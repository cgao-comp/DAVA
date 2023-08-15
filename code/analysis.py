from main import *

args = Args()
print(args.graph_type, args.note)


epoch = 3000
sample_time = 3


def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



for num_layers in range(4,5):
    

    fname_real = args.graph_save_path + args.fname_real + str(0)
    fname_pred = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time)
    figname = args.figure_save_path + args.fname + str(epoch) +'_'+str(sample_time)


    print(fname_real)
    print(fname_pred)


    graph_real_list = load_graph_list(fname_real + '.dat')
    shuffle(graph_real_list)
    graph_pred_list_raw = load_graph_list(fname_pred + '.dat')
    graph_real_len_list = np.array([len(graph_real_list[i]) for i in range(len(graph_real_list))])
    graph_pred_len_list_raw = np.array([len(graph_pred_list_raw[i]) for i in range(len(graph_pred_list_raw))])

    graph_pred_list = graph_pred_list_raw
    graph_pred_len_list = graph_pred_len_list_raw


    

    

    

    

    

    

    

    

    

    

    

    

    




    

    

    

    

    

    

    




    

    

    real_order = np.argsort(graph_real_len_list)[::-1]
    pred_order = np.argsort(graph_pred_len_list)[::-1]
    

    

    graph_real_list = [graph_real_list[i] for i in real_order]
    graph_pred_list = [graph_pred_list[i] for i in pred_order]

    

    

    print('real average nodes', sum([graph_real_list[i].number_of_nodes() for i in range(len(graph_real_list))])/len(graph_real_list))
    print('pred average nodes', sum([graph_pred_list[i].number_of_nodes() for i in range(len(graph_pred_list))])/len(graph_pred_list))
    print('num of real graphs', len(graph_real_list))
    print('num of pred graphs', len(graph_pred_list))


    

    

    

    

    

    

    

    

    

    

    

    

    

    


    

    for iter in range(8):
        print('iter', iter)
        graph_list = []
        for i in range(8):
            index = 32 * iter + i
            

            

            

            graph_list.append(graph_pred_list[index])
            

            print('pred', graph_pred_list[index].number_of_nodes())

        draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_pred')

    

    for iter in range(8):
        print('iter', iter)
        graph_list = []
        for i in range(8):
            index = 16 * iter + i
            

            

            graph_list.append(graph_real_list[index])
            

            print('real', graph_real_list[index].number_of_nodes())
            


        draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_real')












































































































































































