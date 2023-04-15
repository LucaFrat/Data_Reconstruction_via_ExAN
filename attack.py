""" Main file """

import Attack_fig6.datasets as datasets
import Attack_fig6.constants as cons
import Attack_fig6.helpers as helpers
import Attack_fig6.models as models
import Attack_fig6.algorithms as algo


if __name__ == '__main__':

    train_loader, test_loader = datasets.import_dataset('retinamnist')

    if cons.SHOW:
        helpers.data_visualization(train_loader, test_loader)
    
    net_512, optimizer, criterion, layer_dims = models.initialize_network(test_loader)

    for epoch in range(1, cons.N_EPOCHS+1):
        output_training = models.train(net=net_512, 
                                       epoch=epoch, 
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       train_loader=train_loader,
                                       layer_dims=layer_dims
                                        )
        grads, small_grads, params = output_training

        if cons.SHOW:
            helpers.pretty_print_train(*output_training)

    print('\n----  Finished Training  ----\n')

    activation_patterns = algo.attack_nn(grads, params, layer_dims, small_grads)