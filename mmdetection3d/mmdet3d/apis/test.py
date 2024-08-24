import mmcv
import torch
import timeit


def single_gpu_test(model, data_loader, show=False, out_dir=None):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    times = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            start = timeit.default_timer()
            result = model(return_loss=False, rescale=True, **data)
            end = timeit.default_timer()
            duration = end-start
            times.append(duration)
            print(f"Running time: {duration * 1000:.2f} ms")

        if show:
            model.module.show_results(data, result, out_dir)

        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    print(f"Avg runtime: {sum(times)/len(times) * 1000:.2f} ms")
    return results
