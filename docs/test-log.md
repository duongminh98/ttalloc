# Sổ dùng tập kiểm tra

Mỗi lần chạy bất kỳ lệnh nào trên `test_*`, hoặc khóa một cấu hình, ghi một dòng (PLAN mục 7.5).

| Ngày | Lệnh | Tập | Mã băm config | Người duyệt | Ghi chú |
|---|---|---|---|---|---|
| 2026-11-17T18:11:54+00:00 | baseline full | test_main | — | D??ng Tu?n Minh | profile=deer_paper datasets=['math500'] n=500 |

| 2026-11-18 | ??i chi?u CPU t? checkpoint: regrade.py, c?/kh?ng timeout | test_main/math500, ??u ra c? | 3c9fbf102c27 (gi? nguy?n) | D??ng Tu?n Minh ? y?u c?u ??i chi?u c?u h?nh/c?ch ch?m | Kh?ng sinh l?i; 435/500 theo m? DEER tr?n answer_text l?u; grade g?c 441/500; ADR-022 |
