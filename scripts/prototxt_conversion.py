import argparse

def convert_model_prototxt(in_path, out_path, train_lmdb, val_lmdb, batch_size, label_map, test_name_size, num_classes):
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

def generate_solver_prototxt(
    model_path,
    save_path,
    train_size,
    test_size,
    num_epochs=90,
    batch_size=8,
    optimizer="SGD",
    lr=1e-3,
    momentum=0.9,
    weight_decay=5e-4
    ):
    train_steps_per_epoch = train_size / batch_size
    
    solver = {
        "net": model_path,
        "test_iter": test_size // batch_size,
        "test_interval": int(train_steps_per_epoch),
        "max_iter": int(train_steps_per_epoch * num_epochs),

        "display": 100,         # display info every 100 iterations
        "average_loss": 10,     # displayed loss is moving average of 10 samples
        "debug_info": "false",

        "solver_mode": "GPU",
        "device_id": 0,

        "iter_size": 1,
        "type": optimizer,
        "base_lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        
        "lr_policy": "step",    # reduce lr by 10x every 1/3 training schedule
        "gamma": 0.1,
        "stepsize": int(train_steps_per_epoch * num_epochs / 3),

        "snapshot": int(train_steps_per_epoch),     # save snapshot every epoch
        "snapshot_prefix": "./snapshots/snapshot",
        "snapshot_after_train": "true",
        
        "test_initialization": "false",
        "eval_type": "detection",
        "ap_version": "11point"
    }

    with open(save_path, "w") as f:
        for k, v in solver.items():
            f.write(f"{k}: {v}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", type=str)

    parser.add_argument("--in_path", type=str, default="./float/trainval.prototxt")
    parser.add_argument("--out_path", type=str, default="./float/trainval_new.prototxt")
    parser.add_argument("--train_lmdb", type=str, default="/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb")
    parser.add_argument("--val_lmdb", type=str, default="/workspace/SSD/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--label_map", type=str, default="/workspace/SSD/data/labelmap_voc.prototxt")
    parser.add_argument("--test_name_size", type=str, default="/workspace/SSD/data/test_name_size.txt")
    parser.add_argument("--num_classes", type=int, default=21)

    parser.add_argument("--model_path", type=str, default="trainval.prototxt")
    parser.add_argument("--save_path", type=str, default="solver.prototxt")
    parser.add_argument("--train_size", type=int)
    parser.add_argument("--test_size", type=int)
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    args = parser.parse_args()

    if args.action == "model":
        convert_model_prototxt(args.in_path, args.out_path, args.train_lmdb, args.val_lmdb, args.batch_size, args.label_map, args.test_name_size, args.num_classes)
    
    elif args.action == "solver":
        generate_solver_prototxt(args.model_path, args.save_path, args.train_size, args.test_size, args.num_epochs, args.batch_size, args.optimizer, args.lr, args.momentum, args.weight_decay)

    else:
        raise ValueError

if __name__ == "__main__":
    main()
