#from torchvision.models import resnet50, resnet101, vit_b_16, vit_b_32, ViT_B_32_Weights
from torchvision.models import resnet50, resnet101
from torchvision.models import vit_b_16, vit_b_32
from config import CLIPConfig, EchoConfig, CardioConfig
from cardmodel import CardTransformer
#from transformers import AutoTokenizer, AutoModel
import torch
from cardtransform import BaseLiner
from transforms import Numpy2Tensor
import pandas as pd
from math import isnan
from sharednet.models import SharedNet
import sklearn
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
import types
import copy
import pickle

def get_encoders(model_name, pretrained=True, freeze_parameters=False, use_shared_net=True):
  #XPエンコーダ
  print("model_name",model_name)
  if model_name == 'RN50':
    img_encoder = resnet50(pretrained=pretrained)
    img_encoder.fc = torch.nn.Linear(2048, 768)
  elif model_name == 'RN101':
    img_encoder = resnet101(pretrained=pretrained)
    img_encoder.fc = torch.nn.Linear(2048, 768)
  elif model_name == 'ViT-B/32':
    img_encoder = vit_b_32(pretrained=pretrained)
    img_encoder.heads.head = torch.nn.Linear(768, 768)
  elif model_name == 'ViT-B/16':
    img_encoder = vit_b_16(pretrained=pretrained)
    img_encoder.heads.head = torch.nn.Linear(768, 768)

  #心電図エンコーダ
  card_encoder = CardTransformer(out_features=768)

  #共通埋め込み層
  shared_net = SharedNet(out_features=768) if use_shared_net else torch.nn.Identity()

  #freeze parameters
  if freeze_parameters:
    for param in img_encoder.parameters():
      param.requires_grad = False

    if model_name in ['RN50', 'RN101']:
      for param in img_encoder.fc.parameters():
        param.requires_grad = True #Train only final fc layer
    elif model_name in ['ViT-B/32', 'ViT-B/16']:
      for param in img_encoder.heads.head.parameters():
        param.requires_grad = True #Train only final fc layer
    else:
      raise NotImplementedError("NotImplementedError")
      

  return img_encoder, card_encoder, shared_net

import datetime
from dataclasses import dataclass
import torch.nn as nn

def timestamp():
  ct = datetime.datetime.now()
  return ct.strftime("%Y-%m%d %H:%M:%S")


class EchoDataGenerator:
  def __init__(self, max_threshold_svi=100, max_threshold_ee=50, max_threshold_vti=100):
    config = EchoConfig()
    excel_path = config.excel_path
    #self.df = pd.read_excel(excel_path, engine='openpyxl')
    self.df = pd.read_csv(excel_path)
    self.column = config.column
    self.choose_func = config.choose_func
    self.max_threshold_svi = max_threshold_svi
    self.max_threshold_ee = max_threshold_ee
    self.max_threshold_vti = max_threshold_vti
    self.valve_score_dict = config.valve_score_dict

  def fetch(self, patientids):
    return [self.fetch_single(p) for p in patientids]

  def fetch_single(self, patientid):
    extracted = self.df[self.df['Pt_ID']==patientid][self.column]
    scores = [x for x in extracted if not isnan(x)]
         
    if len(scores) == 0:
      return float('nan')
    score = self.choose_func(scores)

    return score

  def get_variables_for_svi_and_ee(self, patientid, examdate=None, choose_func='nearest'):
    extracted = self.df[self.df['Pt_ID']==patientid]
    extracted = extracted.filter(items = ['Exam_ReceptionDateTime', 'Pt_Height', 'Pt_Weight', 'LVOT_VTI', 'LVOT_Diam', 'TMF_E_Velocity', 'TDI_Esep'])
    extracted = extracted.dropna(how='any')
    if len(extracted) == 0:
      return None

    if choose_func=='max':
      extracted = extracted[extracted['LVOT_VTI'] == extracted['LVOT_VTI'].max()]
    elif choose_func=='nearest':
      # 検査日に最も近いデータを選んでくるため、文字列型を日付型に変更
      examdate = datetime.datetime.strptime(examdate, '%Y%m%d')
      extracted['Exam_ReceptionDateTime'] = pd.to_datetime(extracted['Exam_ReceptionDateTime'], format='%Y-%m-%d %H:%M:%S')

      # 前後2週間以内のデータのみを選別
      td = datetime.timedelta(weeks=2)
      extracted = extracted[(extracted['Exam_ReceptionDateTime'] > (examdate - td)) & (extracted['Exam_ReceptionDateTime'] < (examdate + td))]
      if len(extracted) == 0:
        return None

      # XPや心電図検査日との日数差を計算してソート
      extracted['TimeDelta'] = abs(extracted['Exam_ReceptionDateTime'] - examdate)
      extracted = extracted.sort_values(by='TimeDelta')

    #テーブル上最初のデータを取り出す
    extracted = extracted.filter(items = ['Pt_Height', 'Pt_Weight', 'LVOT_VTI', 'LVOT_Diam', 'TMF_E_Velocity', 'TDI_Esep'])
    extracted = extracted.iloc[0].to_list() #temporaly

    return tuple(extracted)
    
  def get_variables_general(self, patientid, target_variables, examdate=None, choose_func='nearest', default_value=np.nan):
    """
    patientidと抜き出したい項目のカラム名を受け取り、抜き出した結果をリストで返す
    """
    extracted = self.df[self.df['Pt_ID']==patientid]
    extracted = extracted.filter(items = ['Exam_ReceptionDateTime'] + target_variables)
    extracted = extracted.dropna(how='any')
    if len(extracted) == 0:
      return [default_value] * len(target_variables)

    if choose_func=='max':
      raise NotImplementedError
    elif choose_func=='nearest':
      # 検査日に最も近いデータを選んでくるため、文字列型を日付型に変更
      examdate = datetime.datetime.strptime(examdate, '%Y%m%d')
      extracted['Exam_ReceptionDateTime'] = pd.to_datetime(extracted['Exam_ReceptionDateTime'], format='%Y-%m-%d %H:%M:%S')

      # 前後2週間以内のデータのみを選別
      td = datetime.timedelta(weeks=2)
      extracted = extracted[(extracted['Exam_ReceptionDateTime'] > (examdate - td)) & (extracted['Exam_ReceptionDateTime'] < (examdate + td))]
      if len(extracted) == 0:
        return [default_value] * len(target_variables)

      # XPや心電図検査日との日数差を計算してソート
      extracted['TimeDelta'] = abs(extracted['Exam_ReceptionDateTime'] - examdate)
      extracted = extracted.sort_values(by='TimeDelta')

    #テーブル上最初のデータを取り出す
    extracted = extracted.filter(items = target_variables)
    extracted = extracted.iloc[0].to_list() #temporaly

    return extracted

  def bmi(self, pid, examdate=None, choose_func='nearest', default_value=np.nan):
    """
    BMI = Pt_Weight(kg) / ((Pt_Height(cm)/100)**2)
    """
    pt_weight, pt_height = self.get_variables_general(pid, ['Pt_Weight', 'Pt_Height'], examdate=examdate, choose_func=choose_func, default_value=default_value)
    if pt_height == 0 or pt_weight == 0:
      return default_value

    bmi = pt_weight / (pt_height/100)**2
    return bmi

  def bmi_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    if examdates is None:
      return [self.bmi(pid, examdate=None, choose_func=choose_func, default_value=default_value) for pid in pids]
    elif examdates is not None:
      return [self.bmi(pid, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def lavi(self, pid, examdate=None, choose_func='nearest', default_value=np.nan):
    """
    BSA = Pt_Height**0.663 * Pt_Weight**0.444 * 0.008883 #体表面積藤本式
    LAVI = LA_LAV / BSA
    """
    pt_weight, pt_height, la_lav = self.get_variables_general(pid, ['Pt_Weight', 'Pt_Height', 'LA_LAV'], examdate=examdate, choose_func=choose_func, default_value=default_value)

    if pt_height == 0 or pt_weight == 0 or la_lav == 0:
      return default_value

    #calculate BSA
    bsa = pt_height**0.663 * pt_weight**0.444 * 0.008883
    if bsa <= 0:
      return default_value

    #calculate LAVI
    lavi = la_lav / bsa
    return lavi

  def lavi_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    if examdates is None:
      return [self.lavi(pid, examdate=None, choose_func=choose_func, default_value=default_value) for pid in pids]
    elif examdates is not None:
      return [self.lavi(pid, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def svi_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    if examdates is None:
      return [self.svi(pid, examdate=None, choose_func=choose_func, default_value=default_value) for pid in pids]
    elif examdates is not None:
      return [self.svi(pid, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def svi(self, patientid, examdate=None, choose_func='nearest', default_value=0):
    """
    BSA = Pt_Height**0.663 * Pt_Weight**0.444 * 0.008883 #体表面積藤本式
    SVI = LVOT_VTI * (LVOT_Diam/2)**2 * pi / BSA

    結果が複数ある場合はPt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_velocity, TDI_EsepがすべてNanでないもののうち、LVOT_VTIが最大のものを使う
    結果が無効な場合（有効な値が1つもない、など）は0を返す
    """
    #Pt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_Velocity, TDI_Esep = self.get_variables_for_svi_and_ee(patientid)
    #extract values
    extracted = self.get_variables_for_svi_and_ee(patientid, examdate=examdate, choose_func=choose_func)
    if extracted is None:
      return default_value
    Pt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_Velocity, TDI_Esep = extracted

    #calculate svi
    BSA = Pt_Height**0.663 * Pt_Weight**0.444 * 0.008883
    if BSA <= 0:
      return default_value

    SVI = LVOT_VTI * (LVOT_Diam/2/10)**2 * np.pi / BSA

    #eliminate unnatural data
    if SVI > self.max_threshold_svi:
      return default_value

    return SVI

  def vti_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    if examdates is None:
      return [self.vti(pid, examdate=None, choose_func=choose_func, default_value=default_value) for pid in pids]
    elif examdates is not None:
      return [self.vti(pid, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def vti(self, patientid, examdate=None, choose_func='nearest', default_value=0):
    """
    BSA = Pt_Height**0.663 * Pt_Weight**0.444 * 0.008883 #体表面積藤本式
    SVI = LVOT_VTI * (LVOT_Diam/2)**2 * pi / BSA

    結果が複数ある場合はPt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_velocity, TDI_EsepがすべてNanでないもののうち、LVOT_VTIが最大のものを使う
    結果が無効な場合（有効な値が1つもない、など）は0を返す
    """
    #Pt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_Velocity, TDI_Esep = self.get_variables_for_svi_and_ee(patientid)
    #extract values
    extracted = self.get_variables_for_svi_and_ee(patientid, examdate=examdate, choose_func=choose_func)
    if extracted is None:
      return default_value
    Pt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_Velocity, TDI_Esep = extracted

    return LVOT_VTI

  def e_e_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    if examdates is None:
      return [self.e_e(pid, examdate=None, choose_func=choose_func, default_value=default_value) for pid in pids]
    elif examdates is not None:
      return [self.e_e(pid, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def e_e(self, patientid, examdate=None, choose_func='nearest', default_value=0):
    """
    E/E' = TMF_E_Velocity / TDI_Esep
    結果が複数ある場合はPt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_velocity, TDI_EsepがすべてNanでないもののうち、LVOT_VTIが最大のものを使う
    結果が無効な場合（有効な値が1つもない、など）は0を返す
    """
    #extract values
    extracted = self.get_variables_for_svi_and_ee(patientid, examdate=examdate, choose_func=choose_func)
    if extracted is None:
      return default_value
    Pt_Height, Pt_Weight, LVOT_VTI, LVOT_Diam, TMF_E_Velocity, TDI_Esep = extracted
    if TDI_Esep <= 0:
      return default_value

    #calculate e_e
    E_E = TMF_E_Velocity / TDI_Esep

    #eliminate unnatural data
    if E_E > self.max_threshold_ee:
      return default_value

    return E_E

  def generalized_batch(self, target, name):
    """
    tagetで指定されたカラム名(target)に該当する項目のデータをバッチで返すインスタンスメソッドを生成する
    インスタンスメソッドを返すインスタンスメソッド
    """
    #self.targetを定義
    self.target = target

    # インスタンスにメソッドを追加
    setattr(self, name, self.method_template)

    # 作成されたメソッド（method_template）を返す
    return getattr(self, name)

  def method_template(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    """
    generalized_batchで使用する基本呼び出し関数
    """
    if examdates is None:
      return [self.get_variables_general(pid, [self.target], examdate=None, choose_func=choose_func, default_value=default_value)[0] for pid in pids]
    elif examdates is not None:
      return [self.get_variables_general(pid, [self.target], examdate=examdate, choose_func=choose_func, default_value=default_value)[0] for pid, examdate in zip(pids, examdates)]

  def valve_generalized_batch(self, target, name):
    """
    tagetで指定されたカラム名(target)に該当する項目のデータをバッチで返すインスタンスメソッドを生成する
    インスタンスメソッドを返すインスタンスメソッド
    弁膜症専用
    """
    # self.targetを定義
    self.target = target

    # インスタンスにメソッドを追加
    setattr(self, name, self.method_template_valve)

    # 作成されたメソッド（method_template）を返す
    return getattr(self, name)

  def method_template_valve(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    ret = []
    for i,pid in enumerate(pids):
      valve_condition = self.get_variables_general(
                              pid, 
                              [self.target], 
                              examdate=None if examdates is None else examdates[i], 
                              choose_func=choose_func, 
                              default_value=default_value)[0]
      if not isinstance(valve_condition, str):
        score = default_value
      elif 'severe' in valve_condition.lower():
        score = self.valve_score_dict['severe'] 
      elif 'mod' in valve_condition.lower():
        score = self.valve_score_dict['moderate']
      elif 'mild' in valve_condition.lower():
        score = self.valve_score_dict['mild']
      else:
        score = self.valve_score_dict['normal']
      ret.append(score)
    return ret

class LabDataGenerator:
  """
  BNPなどの生化学データを取得してくるためのクラス
            PatientID  BirthDate     sex   ExamDate           Hemoglobin  HbA1c(JDSj  ... eGFRcreat   Na    K     BNP NT-proBNP NT-ProBNP*
  0          100054  1937/9/19    male 2012-09-03                 10.7         6.2  ...      28.6  131  4.5  196.69       NaN        NaN
  1          100054  1937/9/19    male 2013-08-20                   11         NaN  ...      27.5  137  4.9  216.07       NaN        NaN
  2          100054  1937/9/19    male 2012-09-24                 10.2         NaN  ...        46  135  3.7  278.69       NaN        NaN
  3          100054  1937/9/19    male 2012-10-02                  9.5         NaN  ...      33.8  131  4.5  304.39       NaN        NaN
  4          100054  1937/9/19    male 2012-01-06                 13.4         7.0  ...      21.3  139  4.4  316.74       NaN        NaN
  ...           ...        ...     ...        ...                  ...         ...  ...       ...  ...  ...     ...       ...        ...
  422938   91000090  2000/3/20  female 2018-11-28                  8.5         NaN  ...       NaN  NaN  NaN     NaN       NaN        NaN
  422939   91000090  2000/3/20  female 2015-03-20                  9.5         NaN  ...       NaN  NaN  NaN     NaN       NaN        NaN
  422940   91000090  2000/3/20  female 2014-05-22                  9.9         NaN  ...       NaN  NaN  NaN     NaN       NaN        NaN
  422941   91000090  2000/3/20  female 2014-01-02  cancelled                   NaN  ...       NaN  NaN  NaN     NaN       NaN        NaN
  422942   91000090  2000/3/20  female 2014-03-10  cancelled                   NaN  ...       NaN  NaN  NaN     NaN       NaN        NaN
  """
  def __init__(self):
    echoconfig = EchoConfig()
    cardconfig = CardioConfig()

    self.df = pd.read_csv(cardconfig.bnp_csv)
    self.df['ExamDate'] = pd.to_datetime(self.df['ExamDate'], format='%Y/%m/%d')

  def general_fetcher(self, pid, df, column_name, examdate=None, choose_func='nearest', default_value=np.nan):
    #PIDで検索
    extracted = df[df['PatientID']==pid]
    extracted = extracted.filter(items = ['ExamDate', column_name])
    extracted = extracted.dropna(how='any')

    #検索対象の日付をdatetime型に変換
    examdate = datetime.datetime.strptime(examdate, '%Y%m%d')
    
    # 前後2週間以内のデータのみを選別
    td = datetime.timedelta(weeks=2)
    extracted = extracted[(extracted['ExamDate'] > (examdate - td)) & (extracted['ExamDate'] < (examdate + td))]
    if len(extracted) == 0:
      return default_value

    # 検査日との日数差を計算してソート
    extracted['TimeDelta'] = abs(extracted['ExamDate'] - examdate)
    extracted = extracted.sort_values(by='TimeDelta')
      

    #テーブル上最初のデータを取り出す
    extracted_value = extracted.iloc[0][column_name] #temporaly

    return extracted_value

  def bnp_prognosis_fetcher(self, pid, df, column_name, examdate=None, choose_func='nearest', default_value=np.nan):
    """
    指定された患者の検査データから、指定日±2週間以内のBNPが100未満である場合に限り、
    指定日以降（同日は含まない）かつ5年以内に実施された検査データのうち、
    BNP値が最も高かった時点の値を取得します。

    パラメータ
    ----------
    pid : str または int
        対象とする患者ID。
    df : pandas.DataFrame
        'PatientID', 'ExamDate', および BNPなどの検査値を含むデータフレーム。
    column_name : str
        取得対象となるカラム名（例: 'BNP'）。
    examdate : str（形式: 'YYYYMMDD'）
        基準となる日付。この日付±2週間以内でBNPが100未満であることが必要。
    choose_func : str
        使用しませんが互換性のため残しています。
    default_value : 任意の型
        条件に合致しない、または有効なデータが見つからなかった場合に返す値。

    戻り値
    -------
    float または default_value
        examdate より後かつ examdate + 5年以内のデータのうち、
        BNP値が最大であったときの値。
        現時点BNPが100以上、または該当データが存在しない場合は default_value を返します。
        なお、BNPの最大値が数値に変換できなかった場合は 0.0 を返します。

    変更点
    -------
    以前の実装では、examdate ±2週間以内でBNP <= 100のデータを基準に、365日以上離れた時点のBNPを返していました。
    現在は、examdate ±2週間以内のBNPが100未満であることを条件とし、
    それ以降3年以内のBNP最大値を返す仕様に変更されています。
    """

    # 該当患者のデータ抽出
    extracted = df[df['PatientID'] == pid].copy()
    extracted = extracted.filter(items=['ExamDate', column_name])
    extracted = extracted.dropna(how='any')

    if len(extracted) == 0:
        return default_value

    # examdateをdatetimeに変換
    examdate = datetime.datetime.strptime(examdate, '%Y%m%d')

    # ±2週間のウィンドウから最も近いBNPを取得
    td = datetime.timedelta(weeks=2)
    windowed = extracted[
        (extracted['ExamDate'] >= examdate - td) &
        (extracted['ExamDate'] <= examdate + td)
    ].copy()

    if windowed.empty:
        return default_value

    windowed['TimeDelta'] = (windowed['ExamDate'] - examdate).abs()
    nearest_row = windowed.loc[windowed['TimeDelta'].idxmin()]
    try:
        nearest_bnp = float(nearest_row[column_name])
    except (ValueError, TypeError):
        return default_value

    if nearest_bnp >= 400:
        return default_value

    # 現時点BNPが100未満だった場合のみ、未来の最大値を評価
    enddate = examdate + datetime.timedelta(days=5 * 365)
    future_data = extracted[
        (extracted['ExamDate'] > examdate) &
        (extracted['ExamDate'] <= enddate)
    ]

    if future_data.empty:
        return default_value

    # 最大BNPを返す、変換失敗時は0.0
    try:
        max_row = future_data.loc[future_data[column_name].astype(float).idxmax()]

        ## bnpを比較するために、検査時点でのbnpを保存する
        #try:
        #  with open("./bnp_current.pkl", "rb") as f:
        #    data = pickle.load(f)
        #except (FileNotFoundError, EOFError):
        #    data = []
        #data.append(nearest_bnp)
        #with open("./bnp_current.pkl", "wb") as f:
        #  pickle.dump(data, f)

        #try:
        #  with open("./pid.pkl", "rb") as f:
        #    data2 = pickle.load(f)
        #except (FileNotFoundError, EOFError):
        #    data2 = []
        #data2.append(pid)
        #print(f"{data2=}")
        #with open("./pid.pkl", "wb") as f:
        #  pickle.dump(data2, f)

        return float(max_row[column_name])
    except (ValueError, TypeError):
        return 0.0


  def bnp_prognosis_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    return [self.bnp_prognosis_fetcher_wrapper(pid, self.df, 'BNP', examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def bnp_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    return [self.general_fetcher_wrapper(pid, self.df, 'BNP', examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]
  
  def K_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    return [self.general_fetcher_wrapper(pid, self.df, 'K', examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]

  def HbA1c_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    return [self.general_fetcher_wrapper(pid, self.df, 'HbA1c(NGSP)', examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]
  #def Hemoglobin_batch(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
  #  return [self.general_fetcher_wrapper(pid, self.df, 'Hemoglobin', examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]
  def generalized_batch(self, target, name):
    """
    tagetで指定されたカラム名に該当する項目のデータをバッチで返すインスタンスメソッドを生成する
    インスタンスメソッドを返すインスタンスメソッド
    """
    # self.targetを定義
    self.target = target

    # インスタンスにメソッドを追加
    setattr(self, name, self.method_template)

    # 作成されたメソッド（name->method_template）を返す
    return getattr(self, name)

  def method_template(self, pids, examdates=None, choose_func='nearest', default_value=np.nan):
    """
    templateとなるmethod
    これが呼び出される前にself.targetが定義されていなければならない
    self.targetには1つしか値を設定できないので、複数のbatch generatorを得たい場合はインスタンスを増やす
    """
    return [self.general_fetcher_wrapper(pid, self.df, self.target, examdate=examdate, choose_func=choose_func, default_value=default_value) for pid, examdate in zip(pids, examdates)]
    

  def general_fetcher_wrapper(self, *args, default_value=np.nan, **kwargs):
    """
    テーブル内の値をそのまま返すパターン（もっと複雑な処理をしたい場合はこの関数だけ書き換えればよい）
    """
    extracted_value = self.general_fetcher(*args, default_value=default_value, **kwargs)
    try:
      extracted_value = float(extracted_value)
    except ValueError:
      extracted_value = default_value
    return extracted_value

  def bnp_prognosis_fetcher_wrapper(self, *args, default_value=np.nan, **kwargs):
    """
    テーブル内の値をそのまま返すパターン（もっと複雑な処理をしたい場合はこの関数だけ書き換えればよい）
    """
    extracted_value = self.bnp_prognosis_fetcher(*args, default_value=default_value, **kwargs)
    try:
      extracted_value = float(extracted_value)
    except ValueError:
      extracted_value = default_value
    return extracted_value
    

def pca_reduce_prime_component(embeds):
  """
  embeds.shape = (sample_number*2, feature_length)
  """
  print(f"{embeds.shape=}")
  pca = PCA()
  pca.fit(embeds)

  feature = pca.transform(embeds)
  print(feature.shape)
  return feature[:,1:]

def specificity_score(y_true, y_pred):
  """
    This function calculates specificity.
    This is copyed from https://qiita.com/player_ppp/items/547afe4b61bee266ea43
  """
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
  return tn / (tn + fp)

def get_metrics_for_test(y_hat, y):
    """
    only for bionary classification!!
    """
    acc = accuracy_score(y, np.round(y_hat))
    precision = precision_score(y, np.round(y_hat))
    recall = recall_score(y, np.round(y_hat))
    specificity = specificity_score(y, np.round(y_hat))
    auc_roc = roc_auc_score(y, y_hat)
    precisions, recalls, _ = precision_recall_curve(y, y_hat)
    auc_pr = auc(recalls, precisions)
    balanced_acc = balanced_accuracy(y, np.round(y_hat))

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'specificity': specificity,
        'auc_pr': auc_pr,
        'balanced_acc': balanced_acc,
    }

def balanced_accuracy(y, y_hat):
    """
    バイナリ配列の正解ラベルと予測ラベルを受け取り、balanced accuracy を計算する関数。

    Args:
      - y (np.array): 正解ラベルのバイナリ配列 (0: 陰性, 1: 陽性)
      - y_hat (np.array): 予測ラベルのバイナリ配列 (0: 陰性, 1: 陽性)

    Returns:
      - float: balanced accuracy (小数点以下3桁で表示)
    """
    # 陰性と陽性のデータ数を取得
    tp = np.sum((y == 1) & (y_hat == 1))
    fn = np.sum((y == 1) & (y_hat == 0))
    tn = np.sum((y == 0) & (y_hat == 0))
    fp = np.sum((y == 0) & (y_hat == 1))

    # TPR (True Positive Rate) と TNR (True Negative Rate) を計算
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Balanced Accuracy の計算
    balanced_acc = (tpr + tnr) / 2

    return balanced_acc

def get_thresholds_and_outcomes(pm_target, echo_config, lab_generator, echo_generator):
    """
    複数ターゲットに対応して thresholds と outcomes を辞書形式で取得する関数

    Parameters
    ----------
    pm_target : str | list[str]
        対象ターゲット名またはそのリスト
    echo_config : object
        echo設定オブジェクト。target_table_path と valve_score_dict を持つこと
    lab_generator : object
        lab系データ生成器。generalized_batch や user_define 関数を持つ
    echo_generator : object
        echo系データ生成器。generalized_batch, valve_generalized_batch 等を持つ

    Returns
    -------
    thresholds : dict[str, float]
        各ターゲット名をキーとした閾値辞書
    outcomes : dict[str, any]
        各ターゲット名をキーとした outcome 辞書
    """

    import pandas as pd

    # ターゲットテーブル読み込み
    exam_target_df = pd.read_csv(echo_config.target_table_path)

    # 単一ターゲットもリストに統一
    targets = pm_target if isinstance(pm_target, (list, tuple)) else [pm_target]

    threshold_dict = {}
    outcome_dict = {}
    lowerthebetter_dict = {}

    for target in targets:
        exam_target = exam_target_df.loc[exam_target_df['name'] == target].iloc[0].to_dict()

        # outcome の取得
        if exam_target['data_type'] == 'lab':
            outcome = lab_generator.generalized_batch(exam_target['column_name'], exam_target['func_name'])
            lab_generator = copy.deepcopy(lab_generator)
        elif exam_target['data_type'] == 'echo':
            outcome = echo_generator.generalized_batch(exam_target['column_name'], exam_target['func_name'])
            echo_generator = copy.deepcopy(echo_generator)
        elif exam_target['data_type'] == 'echo_user_define':
            outcome = getattr(echo_generator, exam_target['func_name'])
        elif exam_target['data_type'] == 'echo_valve':
            outcome = echo_generator.valve_generalized_batch(exam_target['column_name'], exam_target['func_name'])
            echo_generator = copy.deepcopy(echo_generator)
        elif exam_target['data_type'] == 'lab_user_define':
            outcome = getattr(lab_generator, exam_target['func_name'])
        else:
            raise NotImplementedError(f"Unknown data_type: {exam_target['data_type']}")

        # threshold の取得
        if exam_target['data_type'] != 'echo_valve':
            threshold = float(exam_target['threshold'])
        elif exam_target['data_type'] == 'echo_valve':
            threshold = echo_config.valve_score_dict[exam_target['threshold']] - 1
        else:
            raise NotImplementedError

        threshold_dict[target] = threshold
        outcome_dict[target] = outcome
        lowerthebetter_dict[target] = exam_target['lower_the_better']

    return threshold_dict, outcome_dict, lowerthebetter_dict

