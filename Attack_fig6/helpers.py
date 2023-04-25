""" Useful functions """

import numpy as np
import Attack_fig6.constants as cons
import matplotlib.pyplot as plt


def data_visualization(train_loader, test_loader):
    """ Dataset Visualization """

    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    print(f'''
  Length trainset: {len(train_loader)},
  Length testset: {len(test_loader)},
  One test data batch is a tensor of shape: {example_data.shape},
  ''')

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i].permute(1, 2, 0).numpy(), interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i].numpy()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def find_duplicate_indices(r_c, tol=cons.TOL):
    count = 0
    Indices_batch = {}
    for loop_i in range(r_c.shape[0]):
        if r_c[loop_i] != float('inf'):
            min_val = r_c[loop_i] - tol
            max_val = r_c[loop_i] + tol
            indices = np.where((r_c >= min_val) & (r_c <= max_val) & (r_c != float('inf')))
            if len(indices[0]) > 1:
                Indices_batch[count] = indices[0]
                count += 1
                r_c[indices[0]] = float('inf')

    return Indices_batch


def pretty_print_train(grads, small_grads, params):
    print(f'small_grads[0]: \n{small_grads[0]}')
    print(f"small_grads size: {small_grads.shape}")
    print('\nGrad Sizes:')
    for index, key in enumerate(grads.keys()):
        print(f'{key}: {grads[key].size()}')
    print('\nParam Sizes:')
    for index, key in enumerate(grads.keys()):
        print(f'{key}: {params[key].size()}')
