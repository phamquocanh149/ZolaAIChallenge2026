# Baseline PhoGPT trên Kaggle

Pipeline: PhoWhisper-small → PhoGPT-4B-Chat → inspect schema → validate → env.call → kết quả → bước tiếp theo → submission.
Đây là baseline zero-shot, chưa fine-tune. Không gọi API inference bên ngoài, không sửa dữ liệu gốc.

## Cách chạy nhanh

1. Upload `Kaggle_PhoGPT_Baseline.ipynb` lên Kaggle bằng Import Notebook.
2. Add Input dataset của bạn. Notebook mặc định dùng:
   `/kaggle/input/datasets/pqa149/zola-train-public-test/train+public_test`.
   Tự hỗ trợ thêm một tầng `train+public_test/` nếu dữ liệu lồng thư mục.
3. Bật GPU (T4/P100 đều thử được; pipeline chỉ dùng GPU 0), bật Internet để cài dependency và tải model.
4. Chạy các cell theo thứ tự: giải nén source nhúng trong notebook → cài thư viện → kiểm tra data → tải model → smoke test → validation → public.
5. Cell validation/public có cờ `RUN_VALIDATION`, `RUN_PUBLIC`. Đổi thành `True` khi muốn chạy toàn bộ; notebook mặc định chỉ chạy smoke 5 mẫu.
6. Với `TAG="v1"`, file nộp thật nằm tại `/kaggle/working/pipeline_baseline/runs/public_baseline_v1/submission.jsonl`.
   Cell cuối copy thành `/kaggle/working/submission.jsonl` sau khi kiểm tra đủ ID.

Notebook chứa bản sao source, không cần upload riêng Python files. Nếu sửa source, chạy
`python build_notebook.py` để đồng bộ notebook trước khi upload lại.
Không đóng gói hoặc upload lại dataset cuộc thi vào kho công khai.

## Môi trường và model

- PhoGPT-4B-Chat: ~3,7B tham số, instruction/chat, context gốc 8192.
- PhoWhisper-small: ~244M; tổng khoảng 3,94B, dưới giới hạn 6B; LLM dưới 5B.
- Context mặc định 4096, có thể tăng `--context 6144` hoặc `8192` nếu GPU đủ RAM.
- PhoGPT dùng template chính thức `### Câu hỏi: ...\n### Trả lời:` và remote implementation MPT của model.
- Dùng FP16, hoặc BF16 nếu GPU hỗ trợ. Chưa lượng tử hóa; trọng số LLM khoảng 7–8 GB, còn cần ASR, activations và KV cache. Không coi 7–8 GB là toàn bộ VRAM cần dùng.
- `requirements.txt` pin Transformers 4.44.2 để tránh thay đổi API lớn của custom model cũ; giữ PyTorch CUDA do Kaggle cung cấp. Chưa xác nhận chạy GPU trong phiên phát triển local này.
- `download` chỉ là bước chuẩn bị có Internet: tải model và custom Python code, lưu commit SHA vào `models/manifest.json`.
- `infer` ép `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=True`; không tải model lúc inference.
- Mọi output/cache trên Kaggle nằm dưới `/kaggle/working/pipeline_baseline`. Không ghi vào `/kaggle/input`.
- Không cài gì lên máy Windows từ notebook này. Nếu chạy CLI local, luôn đặt `--work-dir` trên ổ E và tự tạo môi trường Python phù hợp.

Để tái sử dụng offline, lưu `models/` thành Kaggle Model/Input riêng, gắn vào notebook và truyền `--llm /kaggle/input/.../llm --asr /kaggle/input/.../asr`.
Các model snapshot phải có đầy đủ `.py`, tokenizer, config và weights. Việc cài dependencies offline cần wheelhouse/container riêng; notebook này chưa phải Docker cuối cùng.

## Chia dữ liệu và đánh giá

`prepare` chỉ đọc train JSONL/các catalog và kiểm tra đường dẫn audio; không đọc environment files.
Khoảng 80/20 development/validation, seed 149. Nhóm câu hiện tại + các câu user trong history trùng sau chuẩn hóa,
rồi chia theo nhóm có/không history, empty/ask/action và số call. Đây là nhóm trùng văn bản chính xác,
chưa phát hiện hết mẫu tương đồng ngữ nghĩa hoặc cùng template. Public không dùng để tạo nhãn, chia tập hay tune.
`splits.json` lưu ID và SHA256 train để phát hiện thay đổi dữ liệu.

Hai chế độ thử nghiệm:

- `--oracle --evaluate --split train --subset validation`: dùng transcript gold để chẩn đoán agent, không phải kết quả end-to-end.
- `--evaluate --split train --subset validation`: ASR cả yêu cầu cuối và history, sau đó chạy agent. Không truyền audio_text hoặc label cho model.

Evaluation riêng đọc snapshot debug qua `env.get_state()` theo README của BTC, ghép các app trong `label.final_state`
để tính state match bằng canonical hash của đúng split. Snapshot và label không vào agent. Không tạo môi trường thứ hai.
CLI không có `--evaluate` cho test.

`metrics.json` chứa tool F1 proxy (maximum bipartite matching), state match, state-gated F1 proxy,
và proxy có xét giới hạn 30 giây. Token-F1 fuzzy dùng lowercase+NFC+whitespace tokens với ngưỡng >0,6.
Đây KHÔNG phải scorer chính thức: Evaluation.txt thiếu công thức và chi tiết tokenization.
State match bỏ qua fuzzy fields theo chính code BTC; fuzzy text vẫn được chấm trong tool-call proxy.

## Agent loop và các giới hạn

Model luôn thấy context, history và danh mục tool được phép. Hành động nội bộ `inspect` đọc schema CÔNG KHAI,
không gọi môi trường, không phải tool nộp. Model cần inspect tool trước khi execute; runner tự đưa schema vào
prompt nếu model gọi tool chưa inspect. Chỉ những `env.call()` thật xuất hiện trong submission.

- Mỗi mẫu tạo đúng một ToolEnv. Không reset, không thử phương án khác trên cùng state.
- Không gọi lại tool trong history vì chúng đã có hiệu lực trong trạng thái ban đầu.
- Validate trước execute; lỗi schema được phản hồi cho model, không biến thành call lỗi đã thực thi.
- Tối đa 12 quyết định (bao gồm inspect), 2 lỗi liên tiếp. Dừng khi ask_missing_info; model được nhắc làm phần độc lập trước.
- Chặn lặp call y hệt khi chưa có write xen giữa; heuristic này có thể chặn yêu cầu lặp hành động hợp lệ và cần phân tích lỗi.
- Prompt không tự cắt history/schema. Vượt context thì ghi `context_overflow`; có thể tăng context hoặc cải thiện routing sau khi đo.
- Greedy ASR/LLM, seed Python/NumPy/Torch, deterministic flags. Một số kernel có cảnh báo không deterministic; cần đo tái lập trên GPU thực.
- ASR xử lý WAV dài theo đoạn 30 giây liên tiếp. Có thể mất từ ở biên đoạn; chưa có overlap/dedup hoặc fine-tune.
- Không cache transcript trong inference: thời gian bao gồm ASR history, không tạo lợi thế giả từ cache offline.

## Giới hạn 30 giây

`--seconds 30` mặc định tính từ trước khi tạo ToolEnv, gồm ASR và vòng agent. Model loading nằm ngoài thời gian từng mẫu.
StoppingCriteria kiểm tra giữa các token; không ngắt được CUDA kernel/prefill đang chạy, nên đây là deadline mềm,
KHÔNG bảo đảm mọi mẫu dưới 30 giây. Trace có `seconds`, `over_30_seconds`, `within_budget`, stop reason.
Mẫu lỗi/timeout vẫn xuất log thực đã chạy và hash hiện tại; không xóa call hoặc dựng hash thay thế.

Smoke oracle trong notebook dùng 120 giây để xem model có thực sự xuất được JSON. Validation/public mặc định 30 giây.
Nếu nhiều timeout/JSON lỗi, không coi pipeline đã đạt yêu cầu nộp final: cần tối ưu số bước, prompt, inference backend
hoặc fine-tune PhoGPT trên development. T4/P100 không tương đương A30 chấm thi.

## CLI tương đương

```bash
python run.py prepare --data-root DATA --work-dir WORK
python run.py download --work-dir WORK
python run.py infer --data-root DATA --work-dir WORK --split train --subset dev --oracle --evaluate --limit 5 --seconds 120 --name smoke_oracle
python run.py infer --data-root DATA --work-dir WORK --split train --subset validation --oracle --evaluate --name val_oracle
python run.py infer --data-root DATA --work-dir WORK --split train --subset validation --evaluate --name val_audio
python run.py infer --data-root DATA --work-dir WORK --split public_test --name public_baseline
python run.py check --data-root DATA --work-dir WORK --split public_test --submission WORK/runs/public_baseline/submission.jsonl
```

Mỗi lệnh infer là process riêng để import đúng `zaic_tools` theo split.
Mỗi `--name` phải mới; không ghi đè hoặc tự resume phiên đã chạy. Nếu thử lại, đổi tên run.
`--limit` chỉ để smoke test: file có ít dòng KHÔNG phải submission đủ public.

`--dry-run` không cần model/GPU, chỉ xuất danh sách call rỗng và hash ban đầu từ ToolEnv.
File có tên `submission.DRY_RUN.jsonl`, chỉ kiểm tra plumbing/format, không phải dự đoán PhoGPT và không nên nộp.
`check` kiểm tra ID/format, không replay để tránh tạo thêm ToolEnv; không khẳng định điểm hoặc tính đúng hành vi.

## Đọc kết quả và cải thiện

`runs/<name>/` có config, model info, submission, traces và metrics (nếu evaluate).
So sánh val_oracle với val_audio để xác định lỗi agent hay ASR. Kiểm tra stop_reason, lỗi JSON, overflow,
tham số ngày/tiền/ID, hỏi thiếu thông tin và call thừa trước khi tăng model hoặc train.
PhoGPT-Chat là model chat tổng quát, chưa được tối ưu riêng cho tool calling; chất lượng zero-shot có thể thấp.
Không dùng public labels hoặc phản hồi public để tạo dữ liệu huấn luyện. Chưa cung cấp fine-tuning hay Docker ở baseline này.

Nguồn model: https://github.com/VinAIResearch/PhoGPT ; https://huggingface.co/vinai/PhoWhisper-small
Thể lệ: `../rule.txt`; chi tiết môi trường: README trong gói dữ liệu.
