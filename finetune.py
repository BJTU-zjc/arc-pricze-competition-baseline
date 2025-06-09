from data import *
from common_stuff import *

async def main():
    # ✅ 配置
    base_path = './data/arc-prize-2024/'
    # ✅ 加载数据
    arc_challenge_file = os.path.join(base_path, 'arc-agi_test_challenges.json')
    arc_solutions_file = os.path.join(base_path, 'arc-agi_training_solutions.json')

    arc_test_set = ArcDataset.from_file(arc_challenge_file)
    if arc_test_set.is_fake:
        arc_test_set.load_replies(arc_solutions_file)

    # ✅ 环境变量、模型参数等
    os.environ["WANDB_DISABLED"] = "true"

    infer_params = dict(min_prob=0.17, store=infer_temp_storage, use_turbo=True)

    # ✅ 后处理（打分 + 提交文件生成）

    for gpu in [0, 1]:
        signal_path = f'{model_temp_storage}_gpu{gpu}_done'
        if os.path.exists(signal_path): os.rmdir(signal_path)

    if arc_test_set.is_fake:
        pass

    # # ✅ 并发启动四个训练/推理任务（作为线程运行）
    # train_proc0 = asyncio.to_thread(start_training, gpu=0)
    # train_proc1 = asyncio.to_thread(start_training, gpu=1)
    # infer_proc0 = asyncio.to_thread(start_inference, gpu=0)
    # infer_proc1 = asyncio.to_thread(start_inference, gpu=1)
    #
    # proc_exit_codes = await wait_for_subprocesses(
    #     train_proc0, train_proc1, infer_proc0, infer_proc1,
    #     print_output=True or arc_test_set.is_fake
    # )
    tasks = [
        run_and_stream(start_training, 0),
        run_and_stream(start_training, 1),
        run_and_stream(start_inference, 0),
        run_and_stream(start_inference, 1),
    ]
    await asyncio.gather(*tasks)
    print("*** All thread tasks completed.")


    # print(f'*** Subprocesses exit codes: {proc_exit_codes}')
    # assert all(x == 0 for x in proc_exit_codes)

    # 推理输出评分与提交
    with RemapCudaOOM():
        model, formatter, dataset = None, MyFormatter(), None
        decoder = Decoder(formatter, arc_test_set.split_multi_replies(), n_guesses=2, frac_score=True).from_store(
            infer_params['store'])
        if use_aug_score or arc_test_set.is_fake:
            decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)

        submission = arc_test_set.get_submission(decoder.run_selection_algo(submission_select_algo))
        with open('submission.json', 'w') as f:
            json.dump(submission, f)

        if arc_test_set.is_fake:
            decoder.benchmark_selection_algos(selection_algorithms)
            with open('submission.json') as f:
                reload_submission = json.load(f)
            print('*** Reload score:', arc_test_set.validate_submission(reload_submission))


# ✅ 启动主函数
if __name__ == "__main__":
    asyncio.run(main())