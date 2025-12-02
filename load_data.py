import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        path = "C:\\Users\\qpalz\\EEG_Dataset\\bcidatasetIV2a-master"
        #path = "C:\\Users\\Kerwin\\EEG_Dataset\\bcidatasetIV2a-master"

        self.data = np.load(f"{path}\\{dataset}")

        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T #'etyp' 存儲事件類型信息
        self.events_position = self.data['epos'].T #'epos' 記錄了每個事件的起始位置
        self.events_duration = self.data['edur'].T #'edur' 記錄了每個事件的持續時間
        self.artifacts = self.data['artifacts'].T #'artifacts' 存儲了人工標記的雜訊或干擾信息

        # Types of motor imagery817000

        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x] #提取所有試驗開始的位置索引

        trials = []
        classes = []

        for index in idxs:
            try:
                type_e = self.events_type[0, index+1] #取得試驗開始位置的下一個事件（index+1），例如：如果 index 位置是 768（試驗開始），index+1 可能是 769（左手）
                class_e = self.mi_types[type_e] #將事件代碼轉換為對應的類別名稱，例如：769 轉換為 'left'
                classes.append(class_e)

                start = self.events_position[0, index] #- - 獲取試驗的開始位置（時間點）
                stop = start + self.events_duration[0, index] #計算試驗的結束位置（時間點）
                trial = self.raw[channel, start:stop] #從指定通道提取這段時間內的腦電信號數據
                trial = trial.reshape((1, -1)) #將試驗數據重塑為二維數組，1 表示一個試驗，-1 表示自動計算另一個維度
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c) #t 獲取該通道的試驗數據，c 獲取對應的類別標籤

            tt = np.concatenate(t, axis=0) #np.concatenate(t, axis=0) 將同一通道的所有試驗數據沿第一個維度連接
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


class BCIDataset(Dataset):
    def __init__(self, args, training=True):
        self.rate = args.test_rate
        self.label_dict = args.label_dict
        self.training = training
        self.batch = args.batch
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

        self.x, self.y = self.load_data(args.subject, args.electrodes)
        minft = self.x.min()
        maxft = self.x.max()
        self.x = ((self.x - minft)/(maxft - minft))


    def load_data(self, subject, electrodes):
        trs,cls = [],[]
        for sub in subject:
            datasets = MotorImageryDataset(f'A0{sub}T.npz')
            trials_, classes_ = datasets.get_trials_from_channels(electrodes)
            trials = np.stack(trials_, axis=1).astype('float32')
            classes = [self.label_dict[la] for la in classes_[0]]

            border = int(len(classes) * (1 - self.rate))

            if self.training == True:
                trs.append(trials[:border])
                cls += classes[:border]
            else:
                trs.append(trials[border:])
                cls += classes[border:]

        trs = np.concatenate(trs,axis=0)

        return trs, cls


    def __len__(self):
        return len(self.x) - self.batch

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --------------------------------------------------------------------------
# 運動想像 (Motor Imagery) EEG 分析流程 (CSP 8-30Hz 版本)
# 方法: 共空間模式 (CSP) + 集成學習分類器
# 數據集: BCI Competition IV-2a
# --------------------------------------------------------------------------

import numpy as np
import mne
import pandas as pd
from matplotlib import pyplot as plt

# Sklearn 相關模組
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from mne.decoding import CSP

# --- 1. 數據載入類 (重構與優化) ---
class MotorImageryDataset:
    """
    用於載入 BCI Competition IV-2a 數據集的類。
    這個版本直接處理並返回 Epochs 陣列，使流程更簡潔。
    """
    def __init__(self, file_path, subject_id):
        """
        初始化數據集。
        :param file_path: 包含 .npz 數據文件的資料夾路徑。
        :param subject_id: 受試者編號 (例如 '1', '2', ... '9')。
        """
        dataset_file = f'A0{subject_id}T.npz'
        full_path = f"{file_path}/{dataset_file}"
        
        try:
            data = np.load(full_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"錯誤：找不到數據文件 {full_path}。請檢查路徑是否正確。")

        self.fs = 250  # 採樣率 (Hz)
        
        # 載入數據，'s' 是 EEG 信號，維度為 (channels, samples)
        self.raw_eeg = data['s'].T
        self.events_type = data['etyp'].ravel()
        self.events_position = data['epos'].ravel()
        self.events_duration = data['edur'].ravel()
        
        # 定義運動想像任務的標籤
        self.mi_labels = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue'}

    def get_epochs(self, tmin, tmax):
        """
        從連續的 EEG 信號中提取所有試驗 (Epochs)。
        返回一個 NumPy 陣列 (trials, channels, samples) 和對應的標籤。
        """
        trial_start_code = 768
        trial_start_indices = np.where(self.events_type == trial_start_code)[0]

        trials = []
        labels = []

        for start_idx in trial_start_indices:
            # 運動想像的提示事件緊跟在試驗開始事件之後
            cue_idx = start_idx + 1
            if cue_idx < len(self.events_type):
                cue_type = self.events_type[cue_idx]
                
                # 檢查這個 cue 是否是我們感興趣的 MI 任務
                if cue_type in self.mi_labels:
                    # 獲取 cue 的時間點作為 epoch 的 0 時刻
                    cue_pos = self.events_position[cue_idx]
                    
                    # 計算 epoch 的開始和結束採樣點
                    start_sample = cue_pos + int(tmin * self.fs)
                    end_sample = cue_pos + int(tmax * self.fs)
                    
                    # 確保索引不越界
                    if end_sample <= self.raw_eeg.shape[1]:
                        epoch_data = self.raw_eeg[:, start_sample:end_sample]
                        trials.append(epoch_data)
                        labels.append(self.mi_labels[cue_type])

        if not trials:
            return np.array([]), np.array([])
            
        # *** 修正點：將數據類型從 float32 改為 float64 以兼容 MNE 濾波器 ***
        return np.stack(trials, axis=0).astype('float64'), np.array(labels)

# --- 2. MNE 濾波器函數 ---
def apply_mne_filter(data, lowcut, highcut, fs):
    """
    使用 MNE 的 filter_data 函數進行濾波。
    :param data: EEG 數據，形狀 (trials, channels, samples)。
    :param lowcut: 低頻截止點。
    :param highcut: 高頻截止點。
    :param fs: 採樣率。
    :return: 濾波後的數據或在出錯時返回 None。
    """
    try:
        # MNE 的濾波器對 NaN/Inf 處理較好，但執行後仍需檢查
        filtered_data = mne.filter.filter_data(data, sfreq=fs, l_freq=lowcut, h_freq=highcut,
                                               method='fir', phase='zero-double',
                                               fir_window='hamming', fir_design='firwin', verbose=False)
        
        # 檢查濾波後是否產生無效值
        if np.any(np.isnan(filtered_data)) or np.any(np.isinf(filtered_data)):
            print(f"      警告: 在濾波 {lowcut}-{highcut} Hz 時產生了 NaN 或 Inf 值。將跳過此頻帶。")
            return None
        return filtered_data
    except Exception as e:
        print(f"      錯誤: MNE 濾波 ({lowcut}-{highcut} Hz) 失敗: {e}。將跳過此頻帶。")
        return None

# --- 3. 主程式設定 ---
class Args:
    # --- 數據路徑與參數 ---
    # !!! 請修改為您的數據集路徑 !!!
    data_path = "C:\\Users\\qpalz\\EEG_Dataset\\bcidatasetIV2a-master" 
    subject_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # subject_list = ['1'] # 可先用單一受試者測試
    
    # --- Epoching 參數 ---
    # 從提示 (cue) 開始後 0.5 秒到 3.5 秒截取數據，共 3 秒
    tmin, tmax = 0.5, 3.5 
    
    # --- CSP 參數 ---
    # *** 這裡是主要修改點：從 FBCSP 改為單一頻帶的 CSP ***
    filter_bands = [(8, 30)] # 使用 8-30Hz 的單一頻帶
    n_csp_components = 4 # 每個類別提取的 CSP component 數量 (OVR策略)

    # --- 特徵選擇與分類 ---
    use_feature_selection = True
    # 因為類別數增加，特徵總數也增加，這裡選擇更多的特徵
    n_features_to_select = 16 
    test_rate = 0.2
    random_state = 42 # 為了結果可重現

    # --- 任務定義 ---
    # 選擇分類任務類型: '4_class' (左/右/腳/舌) 或 '2_class' (左/右)
    #classification_type = '4_class' 
    classification_type = '2_class'

    if classification_type == '4_class':
        class_type = ['left', 'right', 'foot', 'tongue']
        label_dict = {'left': 0, 'right': 1, 'foot': 2, 'tongue': 3}
    elif classification_type == '2_class':
        class_type = ['left', 'right']
        label_dict = {'left': 0, 'right': 1}
    else:
        raise ValueError(f"不支援的分類類型: {classification_type}")

if __name__ == "__main__":
    args = Args()
    results_summary = {} # 用於儲存每個受試者的最終結果

    # --- 4. 主迴圈：處理每個受試者 ---
    for sub_id in args.subject_list:
        print(f"\n{'='*20} 正在處理受試者 A0{sub_id}T {'='*20}")

        # 1. 載入數據並提取 Epochs
        try:
            dataset = MotorImageryDataset(args.data_path, sub_id)
            X_raw, y_raw_labels = dataset.get_epochs(args.tmin, args.tmax)
        except Exception as e:
            print(f"  錯誤: 載入受試者 {sub_id} 的數據失敗: {e}")
            continue

        if X_raw.shape[0] == 0:
            print(f"  警告: 未能為受試者 {sub_id} 載入任何有效的試驗。跳過。")
            continue
        print(f"  成功載入數據，原始 Epochs 形狀: {X_raw.shape}")

        # 2. 過濾感興趣的類別並轉換標籤
        mask = np.isin(y_raw_labels, args.class_type)
        X = X_raw[mask]
        y_labels = y_raw_labels[mask]
        y = np.array([args.label_dict[label] for label in y_labels])

        if X.shape[0] == 0:
            print(f"  警告: 過濾後沒有剩下類別為 {args.class_type} 的試驗。跳過。")
            continue
        print(f"  類別過濾後，Epochs 形狀: {X.shape}, 標籤數量: {len(y)}")

        # 3. 劃分訓練集和測試集
        n_trials = X.shape[0]
        indices = np.arange(n_trials)
        np.random.seed(args.random_state) # 確保每次劃分都一樣
        np.random.shuffle(indices)
        
        border = int(n_trials * (1 - args.test_rate))
        train_idx, test_idx = indices[:border], indices[border:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < len(args.class_type) or len(np.unique(y_test)) < len(args.class_type):
            print(f"  警告: 訓練集或測試集中的類別少於 {len(args.class_type)} 個，可能影響分類性能。")
            print(f"  訓練集類別: {np.unique(y_train)}, 測試集類別: {np.unique(y_test)}")
            if len(np.unique(y_train)) < 2:
                 print("  訓練集類別少於2，無法進行分類，跳過此受試者。")
                 continue

        print(f"  訓練集形狀: {X_train.shape}, 標籤: {len(y_train)} (類別: {np.unique(y_train)})")
        print(f"  測試集形狀: {X_test.shape}, 標籤: {len(y_test)} (類別: {np.unique(y_test)})")

        # 4. CSP 特徵提取
        print("  >> 開始 CSP 特徵提取 (8-30 Hz)...")
        train_features_list = []
        test_features_list = []
        
        # 雖然是單一頻帶，保留迴圈結構以便未來擴展
        for band_idx, (lowcut, highcut) in enumerate(args.filter_bands):
            print(f"    處理頻帶 {band_idx+1}/{len(args.filter_bands)}: {lowcut}-{highcut} Hz")
            
            # 濾波
            X_train_filt = apply_mne_filter(X_train, lowcut, highcut, dataset.fs)
            X_test_filt = apply_mne_filter(X_test, lowcut, highcut, dataset.fs)

            if X_train_filt is None or X_test_filt is None:
                continue

            # CSP (mne 會自動對多分類使用 One-vs-Rest 策略)
            csp = CSP(n_components=args.n_csp_components, reg='ledoit_wolf', log=True, cov_est='epoch')
            try:
                csp.fit(X_train_filt, y_train)
                train_features_list.append(csp.transform(X_train_filt))
                test_features_list.append(csp.transform(X_test_filt))
            except Exception as e:
                print(f"      錯誤: CSP 在頻帶 {lowcut}-{highcut} Hz 失敗: {e}。跳過此頻帶。")
                continue

        if not train_features_list:
            print("  錯誤: CSP 未能提取任何特徵。跳過此受試者。")
            results_summary[f'A0{sub_id}T'] = {'error': 'CSP feature extraction failed'}
            continue

        # 合併所有頻帶的特徵 (此處只有一個頻帶)
        X_train_csp = np.concatenate(train_features_list, axis=1)
        X_test_csp = np.concatenate(test_features_list, axis=1)
        print(f"  CSP 特徵形狀 - 訓練集: {X_train_csp.shape}, 測試集: {X_test_csp.shape}")

        # 5. 特徵選擇 (可選)
        selector = None
        if args.use_feature_selection and X_train_csp.shape[1] > 1:
            print(f"  >> 應用特徵選擇 (SelectKBest, k={args.n_features_to_select})...")
            k = min(args.n_features_to_select, X_train_csp.shape[1])
            selector = SelectKBest(mutual_info_classif, k=k)
            try:
                X_train_final = selector.fit_transform(X_train_csp, y_train)
                X_test_final = selector.transform(X_test_csp)
                print(f"    選擇後特徵形狀 - 訓練集: {X_train_final.shape}, 測試集: {X_test_final.shape}")
            except Exception as e:
                print(f"    錯誤: 特徵選擇失敗: {e}。將使用所有特徵。")
                X_train_final = X_train_csp
                X_test_final = X_test_csp
                selector = None
        else:
            X_train_final = X_train_csp
            X_test_final = X_test_csp
            print("  >> 跳過特徵選擇。")

        # 6. 分類流程
        print("  >> 開始分類流程...")
        
        # 定義分類器
        svm = SVC(probability=True, random_state=args.random_state)
        knn = KNeighborsClassifier()
        lda = LinearDiscriminantAnalysis()

        # 使用投票分類器集成模型
        voting_clf = VotingClassifier(
            estimators=[("svm", svm), ("knn", knn), ('lda', lda)],
            voting='soft'
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', voting_clf)
        ])

        # 7. 超參數搜索 (GridSearchCV)
        param_grid = {
            'classifier__svm__C': [0.1, 1, 10],
            'classifier__svm__kernel': ['rbf', 'linear'],
            'classifier__knn__n_neighbors': [5, 7, 9],
            'classifier__lda__solver': ['svd', 'lsqr']
        }
        
        print("  >> 執行 GridSearchCV 尋找最佳參數...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=0)
        try:
            grid_search.fit(X_train_final, y_train)
        except Exception as e:
            print(f"  錯誤: GridSearchCV 失敗: {e}。跳過分類。")
            results_summary[f'A0{sub_id}T'] = {'error': f'GridSearchCV failed: {e}'}
            continue

        # 8. 使用最佳模型進行評估
        print(f"    最佳參數: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_final)

        # 9. 性能評估與結果儲存
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, 
                                            target_names=args.class_type,
                                            output_dict=True, zero_division=0)

        print(f"\n  受試者 A0{sub_id}T 測試集準確率: {accuracy:.4f}")
        print("  混淆矩陣 (4x4):\n", cm)
        print("  分類報告:\n", classification_report(y_test, y_pred, target_names=args.class_type, zero_division=0))
        
        try:
            cv_scores = cross_val_score(best_model, X_train_final, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean, cv_std = None, None

        results_summary[f'A0{sub_id}T'] = {
            'accuracy': accuracy,
            'cv_mean_score_train': cv_mean,
            'cv_std_score_train': cv_std,
            'report': report_dict,
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
        }

    # --- 5. 總結所有受試者的結果並保存 ---
    print(f"\n{'='*20} 所有受試者結果總結 {'='*20}")

    results_data = {
        'Subject': [],
        'Mean CV Score (Train)': [],
        'Test Accuracy (%)': [],
        'Macro F1 Score (Test %)': []
    }

    for subject, res in results_summary.items():
        if 'accuracy' in res:
            results_data['Subject'].append(subject)
            results_data['Mean CV Score (Train)'].append(res.get('cv_mean_score_train', np.nan))
            results_data['Test Accuracy (%)'].append(res['accuracy'] * 100)
            macro_f1 = res['report'].get('macro avg', {}).get('f1-score', np.nan)
            results_data['Macro F1 Score (Test %)'].append(macro_f1 * 100)
        else:
            print(f"{subject}: 處理失敗 - {res.get('error', '未知錯誤')}")

    if results_data['Subject']:
        results_df = pd.DataFrame(results_data)
        
        # 計算平均值
        mean_row = pd.DataFrame({
            'Subject': ['Average'],
            'Mean CV Score (Train)': [np.nanmean(results_df['Mean CV Score (Train)'])],
            'Test Accuracy (%)': [np.nanmean(results_df['Test Accuracy (%)'])],
            'Macro F1 Score (Test %)': [np.nanmean(results_df['Macro F1 Score (Test %)'])]
        })
        results_df = pd.concat([results_df, mean_row], ignore_index=True)

        # 格式化輸出
        for col in results_df.columns[1:]:
            results_df[col] = results_df[col].round(2)

        print("\n--- CSP (8-30Hz) 四分類結果摘要 ---")
        print(results_df.to_string(index=False))

        # 保存到 CSV
        csv_filename = 'csp_8-30hz_4class_classification_summary.csv'
        results_df.to_csv(csv_filename, index=False, na_rep='N/A')
        print(f"\n結果摘要已保存至 {csv_filename}")
    else:
        print("沒有成功處理的受試者結果可供總結。")
