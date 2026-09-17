\# ISSUE-001：OAK-D 人臉辨識崩潰



\*\*發現日期：\*\* 2026-09-18



\*\*目前狀態：\*\* 未解決



\## 問題描述



執行完整人臉辨識 Pipeline 時，OAK-D 發生 OpenVINO Fatal Error，隨後出現 X\_LINK\_ERROR。



\## 重現指令



`python main.py -id face -fps 2`



\## 錯誤訊息



`Fatal error in openvino 'universal'`



`mvHwOperation 947`



\## 測試紀錄



| 測試                             | 結果         |

| ------------------------------ | ---------- |

| SCRFD 單獨推論                     | 成功，但關閉時曾崩潰 |

| ArcFace 單獨推論                   | 成功         |

| 雙模型平行推論                        | 成功         |

| SCRFD + FrameCropper           | 裁切成功，關閉時崩潰 |

| SCRFD + FrameCropper + ArcFace | 推論期間崩潰     |

| 官方原始程式                         | 相同錯誤       |

| 全新 Python 環境                   | 相同錯誤       |



\## 目前結論



尚無法確定根本原因。



完整串接流程在官方原始程式中也能重現錯誤，因此不能單純歸因於自訂的 Unity TCP 登入功能。



\## 替代方案



嘗試使用 DepthAI Gen2 開源專案，將人臉裁切交由 Python 執行，再送回 OAK-D 提取特徵。



\## 最終解決方式



尚未解決，待後續補充。



