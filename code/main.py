import sys

import create_graphs
from train import *
import processing16
from processing16 import DataReader

import setproctitle
setproctitle.setproctitle('hdp')

torch.cuda.set_device(1)
print('using set cuda!')
target_lambda = torch.Tensor([1.0]).cuda()
if __name__ == '__main__':
    

    args = Args()

    print('File name prefix',args.fname)
    

    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)

    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    

    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    graphs, union_graph_type = create_graphs.create_big(args)

    i = 0
    while True:
        if i >= len(graphs):
            break
        graph = graphs[i]
        if len(graphs[i].edges) < 5:
            del graphs[i]
            

        else:
            i += 1

    if args.small_test:
        small = 100             

        args.batch_ratio = 16   

        i = 0
        while True:
            if i >= len(graphs):
                break
            graph = graphs[i]
            if len(graphs[i].edges) > small:
                del graphs[i]
                

            else:
                i += 1

    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    graphs_validate = graphs[0:int(0.2*graphs_len)]

    graph_validate_len = 0
    for graph in graphs_validate:
        graph_validate_len += graph.number_of_nodes()
    graph_validate_len /= len(graphs_validate)
    print('graph_validate_len', graph_validate_len)

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)



    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))

    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')


    dataset = Graph_DataProcessing(graphs_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    args.max_prev_node = dataset.get_max_prev_node()
    print('max previous node: {}'.format(args.max_prev_node))
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))], 

                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)


    args.hidden_size_rnn_output = 15 if args.feature_numeric is False else 6
    args.num_heads = 4 if args.feature_numeric is False else 2
    args.feature_number = 5 if args.feature_numeric is False else 6


    rnn = GRU_VAE_plain(input_size=args.max_prev_node-1, low_embedding_size=args.low_embedding_size_rnn,
                    GRU_hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                    has_output=True, output_size_to_edge_level=args.hidden_size_rnn_output).cuda()
    output = SimpleAttentionModel(d_key=args.feature_number,
                                  d_query=args.feature_number+args.hidden_size_rnn_output, d_value=args.max_prev_node-1).cuda()
    args.fixed_num = False

    print(rnn)
    print(output)


    train(args, dataset_loader, rnn, output, union_graph_type, graphs)
