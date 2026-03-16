import json
import time
import ntplib
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import sys

# NTP Sync
def get_ntp_time():
    try:
        client = ntplib.NTPClient()
        return client.request('ntp.cnam.fr').tx_time
    except Exception:
        return time.time()

# Min-max normalization
global_max, global_min = {}, {}

def normalize_batch(batch_data):
    normalized = []
    for record in batch_data:
        entry = json.loads(record)
        updated = entry.copy()
        for k, v in entry.items():
            if k.startswith("timestamp") or k == "source_id":
                continue
            global_max[k] = max(global_max.get(k, v), v)
            global_min[k] = min(global_min.get(k, v), v)
            range_val = global_max[k] - global_min[k] or 1
            updated[k] = (v - global_min[k]) / range_val
        normalized.append(updated)
    return normalized

def send_to_kafka(producer, topic, data):
    payload = json.dumps(data).encode("utf-8")
    producer.send(topic, value=payload)
    producer.flush()

def consume_and_preprocess(resource, batch_size, topic_in, topic_out1, topic_out2, ingress, egress, scenario):
    consumer = KafkaConsumer(topic_in, bootstrap_servers=ingress, auto_offset_reset='earliest', enable_auto_commit=False)
    producer = KafkaProducer(bootstrap_servers=egress)
    batch, count = [], 0

    for message in consumer:
        batch.append(message.value.decode("utf-8"))
        count += 1
        if count == batch_size:
            timestamp = get_ntp_time()
            norm = normalize_batch(batch)

            if scenario == "uniform":
                # Uniform split
                split_size = len(norm) // 3
                splits = [norm[i*split_size:(i+1)*split_size] for i in range(2)]
                splits.append(norm[2*split_size:])
            else:
                # 20/30/50 split
                total = len(norm)
                idx1 = int(0.2 * total)
                idx2 = idx1 + int(0.3 * total)
                splits = [norm[:idx1], norm[idx1:idx2], norm[idx2:]]

            for i, data in enumerate(splits):
                send_to_kafka(producer, f"{topic_out1}_{i+1}", data)
                send_to_kafka(producer, f"{topic_out2}_{i+1}", data)

            print(f"[{timestamp}] Sent batches: {[len(s) for s in splits]}")
            batch, count = [], 0

if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: python3 processor_task2.py <polling> <resource> <batch_size> <input_topic> <train_topic> <inf_topic> <in_kafka> <out_kafka> <scenario>")
        sys.exit(1)

    _, _, resource, bsz, topic_in, topic_out1, topic_out2, in_kafka, out_kafka, scenario = sys.argv
    consume_and_preprocess(resource, int(bsz), topic_in, topic_out1, topic_out2, in_kafka, out_kafka, scenario)