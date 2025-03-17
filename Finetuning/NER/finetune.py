import json
import logging
import os
import glob
import re

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import(
    init_logger,
    set_seed,
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_TOKEN_CLASSIFICATION,
    show_ner_report,
    processors,
    f1_pre_rec
)

from src import load_and_cache_examples

from pathlib import Path
proj_root = Path.cwd().parent.parent

tasks_num_labels = 9

logger = logging.getLogger(__name__)

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    # 1. Prepare Training
    
    # 1.1. Create DataLoader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    # 1.2. Set the training steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # 2. Prepare Optimizer and Scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)

    # 3. Load past checkpoint
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # 4. Training Loop

    # 4.1. Setting
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))

    # 4.2. Epoch
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            # 4.3. Train mini-batch
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)

            loss = outputs[0]

            # 4.4. Calculate gradient
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 4.5. Backward and Update Gradient
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                # 5. Evaluation and Save Checkpoint
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step)
                    else:
                        evaluate(args, model, dev_dataset, "dev", global_step)

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    print(f"Model Type: {type(model_to_save)}")
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        mb.write("Epoch {} done".format(epoch + 1))

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step

def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}

    # 1. Set Eval DataLoeader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # 2. Logging eval info
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))

    # 3. Init eval loss, preds
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # 4. Start eval
    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        # 4.1. Generate predictions
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        # 4.2. Save results
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    # 5. Compute loss & metrics
    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    preds = np.argmax(preds, axis=2)

    # 6. Mapping NER label
    labels = ["O", "B-ASP", "I-ASP", "B-OPI", "I-OPI", "B-LOC", "I-LOC", "B-PLC", "I-PLC"]

    label_map = {i: label for i, label in enumerate(labels)}

    # 7. convert preds, out_label_ids to list
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    pad_token_label_id = CrossEntropyLoss().ignore_index

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
     
    # 8. Compute metrics
    result = f1_pre_rec(out_label_list, preds_list)
    results.update(result)

    # 9. Save results
    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))
        logger.info("\n" + show_ner_report(out_label_list, preds_list)) # Show report for each tag result
        f_w.write("\n" + show_ner_report(out_label_list, preds_list))
        
    return results

def main():
    # 1. Load Config
    config_file = "config.json"
    with open(config_file) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    # 2. Set Model and initialize

    # Initialize log and Set random seed
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    # Initialize Processor & get labels
    processor = processors(args)
    labels = processor.get_labels()

        # Load Model Configs
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=tasks_num_labels,
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    # load tokenizer and model
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(
        args.model_name_or_path,
        config=config
    )

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)

    # 3. Load dataset
    args.data_dir = os.path.join(proj_root, args.data_dir, "NER")

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if os.path.join(args.data_dir, args.train_file) else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if os.path.join(args.data_dir, args.test_file) else None
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use testset

        # 4. Train Model
    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    # 5. Evaluate Model
    results = {}
    if args.do_eval:
        checkpoints = list(os.path.dirname(c) for c in
        sorted(glob.glob(args.output_dir + "/**/" + "model.safetensors", recursive=True),
            key=lambda path_with_step: list(map(int, re.findall(r"\d+", path_with_step)))[-1]))
        # 만약 그래도 체크포인트를 못 찾으면 에러 출력      
        if not checkpoints:
            logger.info("No checkpoints found. Trying final trained model instead.")
            checkpoints = [os.path.join(args.output_dir, "final_model")]
        
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # 6. Evaluate at each checkpoint
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            
        # 7. Save the final result
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            if len(checkpoints) > 1:
                for key in sorted(results.keys(), key=lambda key_with_step: (
                        "".join(re.findall(r'[^_]+_', key_with_step)),
                        int(re.findall(r"_\d+", key_with_step)[-1][1:])
                )):
                    f_w.write("{} = {}\n".format(key, str(results[key])))
            else:
                for key in sorted(results.keys()):
                    f_w.write("{} = {}\n".format(key, str(results[key])))

if __name__ == "__main__":
    main()