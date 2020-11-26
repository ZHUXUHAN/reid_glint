import datetime
import mxnet as mx
import numpy as np
from config import cfg

def time():
    return datetime.datetime.now()

def repack():
    save_record = mx.recordio.MXRecordIO(cfg.SAVE_PATH, 'w')

    select_keys = np.arange(0, cfg.TARGET_CLASS, dtype=np.float32)
    select_keys = select_keys * cfg.SKIP_NUM
    record = mx.recordio.MXRecordIO(cfg.READ_PATH, 'r')
    image_count = 0
    pre_time = time()
    while True:
        item = record.read()
        if item is None:
            break
        head, data = mx.recordio.unpack(item)
        class_id = head.label[0]
        if class_id in select_keys:
            label = head.label
            label.flags.writeable = True
            label[0] = class_id / cfg.SKIP_NUM
            transform_head = mx.recordio.IRHeader(head.flag, label, head.id, head.id2)
            assert transform_head.label[0] < cfg.TARGET_CLASS, transform_head.label[0]
            item = mx.recordio.pack(transform_head, data)
            save_record.write(item)
            image_count += 1
            if image_count % 10000 == 0:
                cur_time = time()
                print('time: ', cur_time - pre_time, ' count: ', image_count)

    print(time() - pre_time)
    print("image count: {}".format(image_count))
    record.close()
    save_record.close()


if __name__ == '__main__':
    repack()