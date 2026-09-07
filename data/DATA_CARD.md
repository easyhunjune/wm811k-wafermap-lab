# WM-811K 로컬 데이터 카드

- 상태: 원 배포처·라이선스·로컬 파일 무결성 확인
- 배포처: Kaggle `QINGYI / WM-811K wafer map`
- 배포 URL: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- 원 논문: https://doi.org/10.1109/TSM.2014.2364237
- 다운로드 날짜: 2026-07-28
- 로컬 파일명: `LSWMD.pkl`
- 파일 크기: 2,095,505,977 bytes
- SHA-256: `1d04fccb3dd3176b276878b926b20fead7e077c5751e4d353ea9741a5e7b5c65`
- Kaggle 표시 라이선스: CC0 1.0 Universal (`CC0: Public Domain`)
- 라이선스 원문: https://creativecommons.org/publicdomain/zero/1.0/
- 재사용 조건: CC0 표시에 따라 허가 요청 없이 복제·수정·배포·상업적 이용이
  가능하다. 다만 데이터와 원 논문의 출처를 연구 윤리상 명시하고, 원 저자나
  Kaggle 배포자가 본 프로젝트를 보증하는 것처럼 표현하지 않는다.
- 프로젝트 공개 정책: 라이선스 허용 여부와 별개로 2 GB 원본 pickle은 저장소와
  공개 대시보드에 재배포하지 않는다. 사용자가 위 Kaggle 배포처에서 직접 받도록
  안내한다.

## 로드 후 확인할 스키마

- [x] `waferMap`
- [x] `dieSize`
- [x] `lotName`
- [x] `waferIndex`
- [x] `failureType`
- [x] `trianTestLabel`

## 무결성 메모

- pickle 출처를 신뢰할 수 있는가: 사용자가 위 Kaggle 배포처에서 직접 받았음을
  확인했고, 로컬 SHA-256을 고정했다. pickle은 신뢰한 파일만 역직렬화한다.
- `waferMap` 값이 0, 1, 2로 제한되는가: 예, 전체 811,457행 확인
- 빈 `failureType`과 문자열 `none` 수량: 빈 값 638,507개, `none` 147,431개
- 라벨 9종의 실제 수량: 총 172,950개, `artifacts/splits.metadata.json`에 분할별 기록
- 웨이퍼 맵 감사: 빈 맵 0개, 비정상 차원 0개, 632개 shape, 높이 6~300, 너비 3~205
- 로트 감사: 전체 46,293개, 라벨 데이터 10,762개, 누락 `lotName` 0개
