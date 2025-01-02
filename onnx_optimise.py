""" ONNX optimization script

Run ONNX models through the optimizer to prune unneeded nodes, fuse batchnorm layers into conv, etc.

NOTE: This isn't working consistently in recent PyTorch/ONNX combos (ie PyTorch 2.0.1 and ONNX 1.13),
it seems time to switch to using the onnxruntime online optimizer (can also be saved for offline).

Copyright 2020 Ross Wightman
"""
import argparse
import warnings

import onnx
import onnxoptimizer as optimizer


parser = argparse.ArgumentParser(description="Optimize ONNX model")

parser.add_argument('--model', default='caformer_m36', type=str, metavar='MODEL',
                        choices=['identityformer_m36', 'identityformer_m48', 'identityformer_s12', 'identityformer_s24',
                                'identityformer_s36', 'randformer_s12', 'randformer_s24', 'randformer_s36', 'randformer_m36', 'randformer_m48',
                                'poolformerv2_s12', 'poolformerv2_s24', 'poolformerv2_s36', 'poolformerv2_m36', 'poolformerv2_m48', 'convformer_s18',
                                'convformer_s18_in21k', 'convformer_m36_in21k', 'convformer_m36', 'convformer_b36_in21k', 'convformer_s36_in21k',
                                'convformer_b36', 'convformer_s36', 'convformer_b36_384', 'convformer_m36_384', 'convformer_s18_384', 'convformer_s36_384',
                                'convformer_b36_384_in21ft1k', 'convformer_m36_384_in21ft1k', 'convformer_s18_384_in21ft1k', 'convformer_s36_384_in21ft1k',
                                'convformer_s18_in21ft1k', 'convformer_s36_in21ft1k','convformer_b36_in21ft1k', 'convformer_m36_in21ft1k', 'caformer_m36_384',
                                'caformer_b36', 'caformer_b36_384', 'caformer_m36', 'caformer_s18', 'caformer_s36', 'caformer_s18_384', 'caformer_s36_384',
                                'caformer_b36_in21k', 'caformer_s18_in21k', 'caformer_s36_in21k', 'caformer_m364_in21k', 'caformer_s18_384_in21ft1k',
                                'caformer_m36_384_in21ft1k', 'caformer_s36_384_in21ft1k', 'caformer_b36_384_in21ft1k', 'caformer_s18_in21ft1k',
                                'caformer_s36_in21ft1k', 'caformer_b36_in21ft1k', 'caformer_m36_in21ft1k'],
                        help='Name of model to train')
parser.add_argument("--output", default=None, help="The optimized model output filename")


def traverse_graph(graph, prefix=''):
    content = []
    indent = prefix + '  '
    graphs = []
    num_nodes = 0
    for node in graph.node:
        pn, gs = onnx.helper.printable_node(node, indent, subgraphs=True)
        assert isinstance(gs, list)
        content.append(pn)
        graphs.extend(gs)
        num_nodes += 1
    for g in graphs:
        g_count, g_str = traverse_graph(g)
        content.append('\n' + g_str)
        num_nodes += g_count
    return num_nodes, '\n'.join(content)


def main():
    args = parser.parse_args()

    if args.output == None:
        args.output = f'./{args.model}_optim.onnx'

    args.model = f'./{args.model}.onnx'

    onnx_model = onnx.load(args.model)
    num_original_nodes, original_graph_str = traverse_graph(onnx_model.graph)

    # Optimizer passes to perform
    passes = [
        #'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_dropout',
        'eliminate_nop_pad',
        'eliminate_nop_transpose',
        'eliminate_unused_initializer',
        'extract_constant_to_initializer',
        'fuse_add_bias_into_conv',
        'fuse_bn_into_conv',
        'fuse_consecutive_concats',
        'fuse_consecutive_reduce_unsqueeze',
        'fuse_consecutive_squeezes',
        'fuse_consecutive_transposes',
        #'fuse_matmul_add_bias_into_gemm',
        'fuse_pad_into_conv',
        #'fuse_transpose_into_gemm',
        #'lift_lexical_references',
    ]

    # Apply the optimization on the original serialized model
    # WARNING I've had issues with optimizer in recent versions of PyTorch / ONNX causing
    # 'duplicate definition of name' errors, see: https://github.com/onnx/onnx/issues/2401
    # It may be better to rely on onnxruntime optimizations, see onnx_validate.py script.
    warnings.warn("I've had issues with optimizer in recent versions of PyTorch / ONNX."
                  "Try onnxruntime optimization if this doesn't work.")
    optimized_model = optimizer.optimize(onnx_model, passes)

    num_optimized_nodes, optimzied_graph_str = traverse_graph(optimized_model.graph)
    print('==> The model after optimization:\n{}\n'.format(optimzied_graph_str))
    print('==> The optimized model has {} nodes, the original had {}.'.format(num_optimized_nodes, num_original_nodes))

    # Save the ONNX model
    onnx.save(optimized_model, args.output)


if __name__ == "__main__":
    main()