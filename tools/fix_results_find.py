import csv
import math

INPUT_FILE = "../HistoryRepo/evaluation_results.csv"
OUTPUT_FILE = "evaluation_anomalies.csv"

def is_invalid_accuracy(value):
    """判断 best_accuracy 是否无效"""
    if value is None or value == "":
        return True
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return True
        return False
    except Exception:
        return True


def main():
    invalid_accuracy_rows = []
    failed_status_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames

        for row in reader:
            best_acc = row.get("best_accuracy", "")
            status = row.get("status", "")
            error_msg = row.get("error_message", "")

            # ① best_accuracy 无效
            if is_invalid_accuracy(best_acc):
                invalid_accuracy_rows.append(row)

            # ② 状态异常
            if status != "SUCCESS" or (error_msg and error_msg.strip() != ""):
                failed_status_rows.append(row)

    # 写入输出文件
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_type"] + header)

        for r in invalid_accuracy_rows:
            writer.writerow(["INVALID_ACCURACY"] + [r[h] for h in header])

        for r in failed_status_rows:
            writer.writerow(["STATUS_ERROR"] + [r[h] for h in header])

    print("分析完毕！")
    print(f"无效 best_accuracy 条目数量: {len(invalid_accuracy_rows)}")
    print(f"status / error_message 异常条目数量: {len(failed_status_rows)}")
    print(f"结果已写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
