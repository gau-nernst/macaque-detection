import argparse

def convert_prototxt(in_path, out_path, train_lmdb, val_lmdb, batch_size, label_map, test_name_size, num_classes):
    with open(in_path) as f:
        proto_str = [x.rstrip() for x in f.readlines()]
    
    # data
    proto_str[47] = f"    source: \"{train_lmdb}\""
    proto_str[48] = f"    batch_size: {batch_size}"
    proto_str[134] = f"    label_map_file: \"{label_map}\""
    proto_str[158] = f"    source: \"{val_lmdb}\""
    proto_str[159] = f"    batch_size: {batch_size}"
    proto_str[166] = f"    label_map_file: \"{label_map}\""
    proto_str[4641] = f"    name_size_file: \"{test_name_size}\""

    # box params
    proto_str[4550] = f"    num_classes: {num_classes}"
    proto_str[4615] = f"    num_classes: {num_classes}"
    proto_str[4637] = f"    num_classes: {num_classes}"
    proto_str[4576] = f"      dim: {num_classes}"

    # model architecture
    proto_str[3642] = f"    num_output: {num_classes*8}"
    proto_str[3832] = f"    num_output: {num_classes*6}"
    proto_str[4021] = f"    num_output: {num_classes*6}"
    proto_str[4210] = f"    num_output: {num_classes*6}"
    proto_str[4399] = f"    num_output: {num_classes*4}"

    layer_names = [
        "\"conv4_3_norm_mbox_conf\"",
        "\"fc7_mbox_conf\"",
        "\"conv6_2_mbox_conf\"",
        "\"conv7_2_mbox_conf\"",
        "\"conv8_2_mbox_conf\""
    ]
    for i, x in enumerate(proto_str):
        for name in layer_names:
            if name in x:
                proto_str[i] = f"{proto_str[i][:-1]}_new\""

    with open(out_path, "w") as f:
        for x in proto_str:
            f.write(x)
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, default="./float/trainval.prototxt")
    parser.add_argument("--out_path", type=str, default="./float/trainval_new.prototxt")
    parser.add_argument("--train_lmdb", type=str, default="/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb")
    parser.add_argument("--val_lmdb", type=str, default="/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--label_map", type=str, default="/workspace/SSD/data/labelmap_voc.prototxt")
    parser.add_argument("--test_name_size", type=str, default="/workspace/SSD/data/test_name_size.txt")
    parser.add_argument("--num_classes", type=int, default=21)

    args = parser.parse_args()
    convert_prototxt(args.in_path, args.out_path, args.train_lmdb, args.val_lmdb, args.batch_size, args.label_map, args.test_name_size, args.num_classes)

if __name__ == "__main__":
    main()
