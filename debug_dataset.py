from dataclasses import asdict

from models import build_model_and_tokenizer, parse_args
from data import build_concat_train_dataset, build_eval_dataset_dict, get_data_collator, get_compute_metrics_dict
from engine import TrainerWithGenToEval

def train():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args))

    args.gradient_checkpointing_kwargs = {'use_reentrant': False}
    # If only one eval set is provided, unwrap it so Trainer gets a callable compute_metrics
    if eval_dataset_dict and len(eval_dataset_dict) == 1:
        eval_name, eval_dataset = next(iter(eval_dataset_dict.items()))
        eval_compute_metrics = compute_metrics_dict[eval_name]
    else:
        eval_dataset = eval_dataset_dict
        eval_compute_metrics = compute_metrics_dict

    # nv_example = train_dataset[1]
    item_1 = train_dataset[2]
    print(item_1)

    # trainer = TrainerWithGenToEval(
    #     model=model, tokenizer=tokenizer,
    #     args=args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     data_collator=data_collator,
    #     compute_metrics=eval_compute_metrics,
    # )
    # trainer.train()
    # trainer.save_model()

    # if eval_dataset_dict is not None and len(eval_dataset_dict) != 1:
    #     metrics = {}
    #     for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
    #         trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
    #         metrics.update(
    #             trainer.evaluate(
    #                 eval_dataset=eval_dataset,
    #                 metric_key_prefix=f"eval_{eval_dataset_name}",
    #             )
    #         )
    #     print(metrics)

if __name__ == "__main__":
    train()
