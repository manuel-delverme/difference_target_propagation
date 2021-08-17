dataset = str('mnist')  # ['mnist', 'cifar10'], help='Used dataset for classification/regression. Default: %(default)s.')
epochs = int(100)  # 'Number of training epochs. ' + 'Default: %(default)s.')
batch_size = int(100)  # 'Training batch size. Choose divisor of "num_train". Default: %(default)s.')
lr = .01
lr_fb = 0.000101149118237  # 'Learning rate of optimizer for the feedback parameters. Default: %(default)s.')
target_stepsize = float(0.01)  # 'Step size for computing the output target basedon the output gradient. Default: %(default)s.')
optimizer = str('Adam')  # ['SGD', 'RMSprop', 'Adam'], help='Optimizer used for training. Default: %(default)s.')
momentum = float(0.0)  # 'Momentum of the SGD or RMSprop optimizer. Default: %(default)s.')
sigma = float(0.08)  # 'svd of gaussian noise used to corrupt the hidden layer activations for computing the reconstruction loss.')

log_interval = None  # 'Each <log_interval> batches, the batch resultsare logged to tensorboard.')

forward_wd = float(0.0)  # help='Weight decay for the forward weights. Default: %(default)s.')
feedback_wd = float(0.0)  # help='Weight decay for the feedback weights. Default: %(default)s.')
epochs_fb = int(1)  # , help='Number of training epochs. Default: %(default)s.')
soft_target = float(0.9)
norm_ratio = float(1)  # help='hyperparameter used for computing the minimal ' 'norm update of the parameters, given the targets.
extra_fb_epochs = int(0)  # help='After each epoch of training, the fb parameterswill be trained for an extra extra_fb_epochs epochs')
extra_fb_minibatches = int(
    0)  # After each minibatch training of the forward parameters,we do <N> extra minibatches trainingfor the  feedback weights. The extra minibatches randomly sampled from the trainingset')
gn_damping_training = float(0.)  # help='Thikonov damping used to train the GN network.')
loss_scale = float(1.)

beta1 = float(0.99)  # help='beta1 training hyperparameter for the adam optimizer. Default: %(default)s')
beta2 = float(0.99)  # help='beta2 training hyperparameter for the adam optimizer. Default: %(default)s')
epsilon = 1e-4  # help='epsilon training hyperparameter for the adam optimizer. Default: %(default)s')
beta1_fb = 0.99  # help='beta1 training hyperparameter for the adam feedback optimizer. Default: %(default)s')
beta2_fb = 0.99  # help='beta2 training hyperparameter for the adam feedback optimizer. Default: %(default)s')
epsilon_fb = 1e-4  # help='epsilon training hyperparameter for the adam feedback optimizer. Default: %(default)s')

hidden_layers = None
num_hidden = int(2)  # 'Number of hidden layer in the ' + 'network. Default: %(default)s.')
size_hidden = 256  # 'Number of units in each hidden layer of the ' + '(student) network. If you provide a list, you can have layers of different sizes
size_input = int(784)  # 'Number of units of the input. Default: %(default)s.')
size_output = int(10)  # 'Number of units of the output. Default: %(default)s.')
size_hidden_fb = int(500)  # 'Number of units of the hidden feedback layer (in the DDTP-RHL variants).. Default: %(default)s.')
hidden_activation = 'tanh'  # ['tanh', 'relu', 'linear', 'leakyrelu', 'sigmoid'], help='Activation function used for the hidden layers. Default: $(default)s.')
network_type = ['DTP', 'LeeDTP'][1]  # help='Variant of TP that will be used to train the network. See the layer classes for explanations of the names. Default: %(default)s.')
random_seed = int(42)  # 'Random seed. Default: %(default)s.')

classification = True
output_activation = 'softmax'

fb_activation = hidden_activation  # Activation function used for the feedback targets for the hidden layers. Default the same as hidden_activation.
hidden_fb_activation = hidden_activation  # Activation function used for the hidden layers of the direct feedback mapping.

optimizer_fb = optimizer  # [None, 'SGD', 'RMSprop', 'Adam'], help='Optimizer used for training the feedback parameters.')

# Manipulating command line arguments if asked
size_mlp_fb = None
gn_damping = 0.
use_cuda = True
device = "cuda" if use_cuda else "cpu"
