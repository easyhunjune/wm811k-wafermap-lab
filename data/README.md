# 로컬 데이터 위치

Kaggle의 [QINGYI / WM-811K wafer map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
에서 `LSWMD.pkl`을 받아 이 폴더에 둔다. Kaggle 페이지의 표시 라이선스는
[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)이다.
데이터 파일은 Git이나 공개 대시보드에 포함하지 않는다.

파일을 받은 뒤 `DATA_CARD.md`를 만들고 다음을 기록한다.

```text
배포처: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
원 논문:
다운로드 날짜:
파일명:
파일 크기:
SHA-256:
라이선스/재배포 조건: CC0 1.0 Universal
스키마 확인 결과:
```

PowerShell에서 SHA-256 확인:

```powershell
Get-FileHash .\data\LSWMD.pkl -Algorithm SHA256
```
