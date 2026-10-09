# FedFM

Tài liệu này mô tả luồng FedFM hiện có trong repository và các vấn đề logic đã được rà soát. Đây là ghi nhận theo mã nguồn hiện tại; các lỗi bên dưới chưa được sửa.

## Tổng quan

Entry point là `exps/federated_main.py`. Mã hiện tại hướng tới bài toán ảnh X-quang COVID-19 với bốn lớp:

| Nhãn | Thư mục dữ liệu | Chỉ số |
| --- | --- | ---: |
| COVID | `COVID` | 0 |
| Lung opacity | `Lung_Opacity` | 1 |
| Normal | `Normal` | 2 |
| Viral pneumonia | `Viral Pneumonia` | 3 |

Mỗi client có một encoder ResNet-50 tạo vector 768 chiều và một MLP phân loại bốn lớp. MLP trả về log-probability; encoder trả thêm embedding đã chuẩn hóa L2 để tính prototype loss. Loss local là tổng của NLL phân loại và `MultiProtoSupConLoss`.

Ở cuối mỗi vòng, code trung bình các tham số có tên chứa `head` trên tất cả client rồi chép head trung bình lại cho từng client. Encoder của từng client vẫn riêng. Mặc dù tên lớp encoder là `MEDCLIPVisionModel`, ResNet-50 được tạo với `pretrained=False` và không có checkpoint pretrained nào được nạp trong luồng này.

## Dữ liệu

`--data_dir` được kỳ vọng trỏ trực tiếp tới thư mục chứa các thư mục lớp, ví dụ:

```text
data_dir/
├── COVID/images/*
├── Lung_Opacity/images/*
├── Normal/images/*
└── Viral Pneumonia/images/*
```

`lib/covid19.py` sắp xếp đường dẫn ảnh theo tên file và gán nhãn theo thứ tự bảng trên. Validation lấy 1/6 đầu của mỗi lớp. Hiện tại train lại lấy **toàn bộ** ảnh của mỗi lớp, bao gồm cả validation.

## Chạy

Các dependency Python không được khai báo trong repository. Cần cài PyTorch, torchvision, NumPy, Pillow, tqdm và tensorboardX trong môi trường chạy.

Ví dụ gọi entry point cho dataset hiện được hỗ trợ:

```bash
python -m exps.federated_main --dataset covid19 --data_dir /path/to/data --num_classes 4
```

`--num_classes` nên là `4` để khớp với classifier và prototype loss đang hard-code bốn lớp. Lưu ý: câu lệnh trên chỉ minh họa giao diện hiện tại; do các lỗi được liệt kê bên dưới, kết quả huấn luyện hiện chưa phản ánh đúng thiết lập federated mong đợi.

## Luồng huấn luyện hiện tại

1. `get_dataset` tạo train/validation và chia chỉ số train thành các nhóm client bằng `covid19_iid`.
2. Tạo một `ClientModel` khởi tạo độc lập cho mỗi client.
3. Mỗi vòng, từng client chạy local update, tính NLL và prototype loss, rồi ghi loss/accuracy vào TensorBoard.
4. Trung bình head weights giữa các client và cập nhật lại head của từng client.

Prototype được đọc từ đường dẫn cố định `/kaggle/working/MyProj/exps/global_protos_dict.pt`; nếu file không tồn tại thì loss prototype bằng 0. Repository có `exps/global_protos_dict.pt`, nhưng entry point không đọc file này. Trong code hiện tại không có bước cập nhật hoặc lưu prototype sau local training. Các tham số `ways`, `shots`, `stdev` và `k_list` không điều khiển cách chia dữ liệu hiện tại.

## Phát hiện logic cần xử lý

### Nghiêm trọng

1. **Client không train trên partition riêng.** `LocalUpdate.train_val_test` nhận `idxs` nhưng tạo `DataLoader(dataset, ...)` thay vì `DataLoader(DatasetSplit(dataset, idxs_train), ...)`. Vì vậy mọi client train trên toàn bộ train set và `user_groups` không có tác dụng. Xem `lib/update.py`.
2. **Train/validation bị rò rỉ dữ liệu.** Dataset train lấy toàn bộ ảnh, trong khi validation lấy 1/6 đầu của chính các ảnh đó. Đánh giá validation vì vậy không độc lập. Xem `lib/covid19.py`.
3. **Prototype toàn cục không được cập nhật.** `global_protos` chỉ được load một lần trước vòng lặp; `local_protos` được tạo nhưng không dùng. Nếu không có file prototype ở đúng đường dẫn Kaggle thì prototype loss luôn bằng 0; nếu có thì mọi vòng dùng nguyên prototype cũ.

### Nghiêm trọng khi chạy cấu hình mặc định / môi trường khác

4. **Dataset mặc định không chạy được.** Parser mặc định `--dataset mnist`, nhưng `k_list` chỉ được khởi tạo trong nhánh `covid19`; sau đó `print(k_list)` gây `UnboundLocalError`. `get_dataset` cũng chỉ khởi tạo dataset trong nhánh `covid19`, nên các dataset khác không được hỗ trợ thực tế.
5. **Đường dẫn prototype phụ thuộc môi trường Kaggle.** File được tìm ở đường dẫn tuyệt đối, không phải file `exps/global_protos_dict.pt` có trong repository. Chạy nơi khác sẽ âm thầm khởi tạo dict rỗng sau cảnh báo, khiến nhánh prototype không hoạt động như kỳ vọng.
6. **Có thể lỗi nếu loader không tạo batch nào.** Training và validation đều đặt `drop_last=True`. Nếu số mẫu nhỏ hơn batch size, các danh sách loss rỗng dẫn đến chia cho 0; training cũng trả về `acc_val` chưa được gán. Validation có thể có ít hơn 32 mẫu sau khi chia.

### Sai lệch phương pháp / kết quả

7. **Encoder không dùng pretrained weights.** Tên `MEDCLIPVisionModel` có thể gây hiểu nhầm: ResNet-50 được khởi tạo ngẫu nhiên (`pretrained=False`). Encoder còn không được đồng bộ giữa client.
8. **Tham số `frac` không được áp dụng.** Mỗi vòng luôn cập nhật toàn bộ client (`np.arange(args.num_users)`).
9. **Validation dùng augmentation ngẫu nhiên.** Cùng transform train (flip, jitter, affine, crop) được dùng cho validation, làm metric thay đổi ngẫu nhiên giữa các lượt.
10. **Không có đánh giá cuối cùng trong entry point.** Hàm đánh giá cuối bị comment; `FedFM` cũng không trả model/metric và không lưu checkpoint. Các con số hiện ghi là local train accuracy và validation log trong từng local update.

## Ghi chú rà soát

Đây là rà soát tĩnh theo luồng được gọi từ `exps/federated_main.py`. Không chạy training hay test suite trong lần rà soát này. Những hàm đánh giá/helper còn lại trong `lib/update.py` chưa được xác nhận là tương thích với luồng FedFM chính.
