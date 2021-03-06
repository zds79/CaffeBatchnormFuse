#!/usr/bin/python
#
# Brief:
#   Do the preprocess for converting of the caffe model to Nano.AI model.
#
#   1. Fuse the Conv-BatchNorm-Scale layer group and Batchnorm-Scale layer group in caffe model
#   to speed up the inference
#   2. Remove the dropout layer
#   3. Update the in-place layer to make sure the bottom and top blob has different name
#
# Reference:
#   https://github.com/zhang-xin/CNN-Conv-BatchNorm-fusion
#   https://github.com/hmph/caffe-fold-batchnorm
#
# Author:
#   Dongsheng Zhang, Michael Wang, Raymond Lei
#

from __future__ import print_function
import argparse
import numpy as np
import sys,os
# suprress Caffe verbose prints
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

np.set_printoptions(suppress=True)

class CaffeBatchnormFuse:
    def __init__(self, network, model):
        self.network = network
        self.model = model
        caffe.set_mode_cpu()
        self.orig_net = caffe.Net(network, model, caffe.TEST)
        self.net = self.orig_net
        self.updated_proto = self.network.replace('.prototxt', '_m.prototxt')
        self.updated_model = self.model.replace('.caffemodel', '_m.caffemodel')

    # Get the producer layer of the specified blob
    # search from the bottom layer to the top layer (reverse)
    #
    # For the in-place layer, the bottom blob has the same name of the top blob
    # we need to find the last layer(producer) of this blob.
    def get_blob_producer_layer(self, proto, blob_name, layer_name):
        start_check = 0
        i = len(proto.layer)
        while i > 0:
            i -= 1
            if proto.layer[i].name == layer_name:
                start_check = 1
                continue

            if start_check == 1:
                top_blobs = len(proto.layer[i].top)
                while top_blobs > 0:
                    top_blobs -= 1
                    if proto.layer[i].top[top_blobs] == blob_name:
                        return proto.layer[i]

        return None

    #
    # Get the consumer layers of the specified blob
    # search from the top layer to the bottom layer (reverse)
    #
    def get_blob_consumer_layer(self, proto, blob_name, layer_name):
        consumer_layers = []
        start_check = 0
        layer_len = len(proto.layer)
        i = 0
        while i < layer_len:
            if proto.layer[i].name == layer_name:
                start_check = 1
                i += 1
                continue

            if start_check == 1:
                bottom_blobs = len(proto.layer[i].bottom)
                while bottom_blobs > 0:
                    bottom_blobs -= 1
                    if proto.layer[i].bottom[bottom_blobs] == blob_name:
                        consumer_layers.append((i,proto.layer[i]))
                        #print("  #consumer layer: [{}] [{}] [{}]".format(i, proto.layer[i].type, proto.layer[i].name))
                        break

            i += 1

        return consumer_layers

    #
    # Re-name the top blob name if the bottom blob has the same name
    #
    # For the in-place layers, the bottom blob may has the same name as the top blob.
    #
    # We need to do the quantization for each layer based on the blob, the same bottom and
    # top blob name will introduce the issue while get the blob data of each layer.
    #
    # So, we need to re-name the top to the layer's name to avoid this issue.
    #
    def update_same_top_bottom_name_layer(self, proto):
        # get layer list which has the same top and bottom blob name
        layer_has_same_top_bottom_name_list = []
        for i in range(len(proto.layer)):
            layer = proto.layer[i]
            if str(layer.type).lower() in ['data','input','eltwise', 'dropout']:
                continue
            if layer.top[0] == layer.bottom[0]:
                layer_has_same_top_bottom_name_list.append((i,layer))

        #print("#bottom same as top blob layers: {}#".format(len(layer_has_same_top_bottom_name_list)))
        # Update all layers has the same top and bottom blob
        for item in reversed(layer_has_same_top_bottom_name_list):
            (idx,layer) = item
            #print("#[{}] [{}] [{}] [{}]".format(idx, layer.type, layer.name, layer.top[0]))
            # update all the consumer layers
            for l in proto.layer[idx + 1:]:
                for j in range(len(l.bottom)):
                    if l.bottom[j] == layer.top[0]:
                        #print('layer:{} update bottom blob name {} to {}'.format(l.name, l.bottom[j], layer.name))
                        l.bottom[j] = layer.name
            layer.top[0] = layer.name

    # get the conv layer name of the conv-batchnorm-scale group
    def get_fuse_conv_layer(self, proto, layer):
        if layer.type == u'Scale':
            bottom_layer = self.get_blob_producer_layer(proto, layer.bottom[0], layer.name)
            if bottom_layer and bottom_layer.type == u'BatchNorm':
                bottom2_layer = self.get_blob_producer_layer(proto, bottom_layer.bottom[0], bottom_layer.name)
                if bottom2_layer and bottom2_layer.type == u'Convolution':
                    return bottom2_layer
        elif layer.type == u'BatchNorm':
            bottom_layer = self.get_blob_producer_layer(proto, layer.bottom[0], layer.name)
            if bottom_layer and bottom_layer.type == u'Convolution':
                return bottom_layer
        return None

    # get layer index with layer name
    def get_layer_index(self, proto, name):
        for i in range(len(proto.layer)):
            if proto.layer[i].name == name:
                return i
        return -1

    # do the conv-batchnorm-scale fusion
    def fuse(self):
        # batch norm layer name list in the conv-batch-scale groupt
        cv_bn_sc_group_layers_bn = {}
        # scale layer name list in the conv-batch-scale groupt
        cv_bn_sc_group_layers_sc = {}
        # scale layer name list in the batch-scale groupt
        bn_sc_group_layers_sc = {}
        # add a dic to read the batchNorm eps parameter instead of using default ones
        group_layers_bn_eps = {}
        proto = caffe_pb2.NetParameter()
        text_format.Merge(open(self.network).read(), proto)

        # Find the Conv-BN-Scale groups
        remove_layer_list = []
        for i in range(len(proto.layer)):
            layer = proto.layer[i]
            #print("{} {} {} bottom:{}".format(i, layer.name, layer.type, layer.bottom[0]))
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
                conv_layer = self.get_fuse_conv_layer(proto, layer)
                if conv_layer:
                    if layer.type == u'BatchNorm':
                        print("[Fusion:]\r\n  cv:{}\r\n  sc:{}".format(conv_layer.name, layer.name))
                        cv_bn_sc_group_layers_bn[conv_layer.name] = layer.name
                        # read the layer name and the correspond eps value
                        group_layers_bn_eps[conv_layer.name] = layer.batch_norm_param.eps
                    else:
                        cv_bn_sc_group_layers_sc[conv_layer.name] = layer.name
                        print("  sc:{}".format(layer.name))
                        # update the bias term
                        conv_layer.convolution_param.bias_term = True
                        # Need to remove the batchnorm and the scale layer later, so we need
                        # to update the net's topology, connect conv's top to scale's top
                        #
                        # Also, as we won't change the .model, we need to rename the conv's name and the conv's top
                        # name else will cause the load error(updated .proto but old .model)
                        #
                        # Add "_m" suffix to the conv's layer name. 
                        # Set the conv's top name to the scale's top name.
                        # Add the "_m" suffix to the blob of all the consumer layers from conv layer.
                        conv_layer_index = self.get_layer_index(proto, conv_layer.name)
                        if conv_layer_index >= 0:
                            orig_top_name = layer.top[0]
                            updated_top_name = orig_top_name + '_m'
                            conv_layer.top[0] = updated_top_name
                            conv_layer.name += '_m'
                            for l in proto.layer[conv_layer_index:]:
                                #print("  #check layer {} {} tops: {} bottoms:{}".format(l.name, l.type, l.top, l.bottom))
                                for j in range(len(l.top)):
                                    if l.top[j] == orig_top_name:
                                        #print('    layer:{} update top blob name: {}->{}'.format(l.name, l.top[j], updated_top_name))
                                        l.top[j] = updated_top_name
                                for k in range(len(l.bottom)):
                                    if l.bottom[k] == orig_top_name:
                                        #print('    layer:{} update bottom blob name: {}->{}'.format(l.name, l.bottom[j], updated_top_name))
                                        l.bottom[k] = updated_top_name
                    #print("  #add layer {} to remove list".format(layer.name))
                    remove_layer_list.append(layer)
                elif layer.type == u'BatchNorm':
                    if (len(layer.bottom) != 1) or (len(layer.top) != 1):
                        print("ERROR!Expected bn layer to have one top and bottom")
                    prev_layer, next_layer = proto.layer[i-1], proto.layer[i+1]
                    if not (prev_layer.top == layer.bottom and layer.top == next_layer.bottom):
                        print("ERROR!Could not find previous and next nodes for batch norm layer")
                    if next_layer.type != 'Scale':
                        print("ERROR!Expected Scale layer to follow batch norm layer")
                    if not (len(prev_layer.top) == 1 and len(next_layer.bottom) == 1):
                        print("ERROR!Expected previous and next blobs to have only one input and output")
                    # read the layer name and the correspond eps value
                    bn_sc_group_layers_sc[next_layer.name] = layer.name
                    group_layers_bn_eps[next_layer.name] = layer.batch_norm_param.eps
                    print("[Fusion:]\r\n  bt:{}\r\n  sc:{}".format(layer.name, next_layer.name))
                    next_layer.name += "_f"
                    next_layer.bottom[0] = prev_layer.top[0]
                    remove_layer_list.append(layer)

        # remove the BN and Scale in the Conv-BN-Scale group
        for l in remove_layer_list:
            proto.layer.remove(l)

        self.update_same_top_bottom_name_layer(proto)

        # updated_proto = self.network.replace('.prototxt', '_m.prototxt')
        # updated_model = self.model.replace('.caffemodel', '_m.caffemodel')

        # save the updated network topology .proto
        with open(self.updated_proto, 'w') as f:
            f.write(str(proto))

        # calc new conv weights from original conv/bn/sc weights
        conv_new_w = {}
        conv_new_b = {}
        #print("###conv_layers_bn:{}###".format(len(group_layers_bn)));
        index = 0
        for layer in cv_bn_sc_group_layers_bn:
            #print("++conv_layers_bn:{}:{}++".format(index, layer))
            index += 1
            old_w = self.orig_net.params[layer][0].data
            if len(self.orig_net.params[layer]) > 1:
                old_b = self.orig_net.params[layer][1].data
            else:
                old_b = np.zeros(self.orig_net.params[layer][0].data.shape[0],
                                 self.orig_net.params[layer][0].data.dtype)
            if self.orig_net.params[cv_bn_sc_group_layers_bn[layer]][2].data[0] != 0:
                s = 1 / self.orig_net.params[cv_bn_sc_group_layers_bn[layer]][2].data[0]
            else:
                s = 0
            u = self.orig_net.params[cv_bn_sc_group_layers_bn[layer]][0].data * s
            v = self.orig_net.params[cv_bn_sc_group_layers_bn[layer]][1].data * s
            alpha = self.orig_net.params[cv_bn_sc_group_layers_sc[layer]][0].data
            beta = self.orig_net.params[cv_bn_sc_group_layers_sc[layer]][1].data
            # use the eps value read from the prototxt instead of the default value
            eps = group_layers_bn_eps[layer]
            conv_new_b[layer] = alpha * (old_b - u) / np.sqrt(v + eps) + beta
            conv_new_w[layer] = (alpha / np.sqrt(v + eps))[...,
                                                       np.newaxis,
                                                       np.newaxis,
                                                       np.newaxis] * old_w

        # calc new scale weights from original batchNorm&Scale weights
        scale_new_w = {}
        scale_new_b = {}
        for layer in bn_sc_group_layers_sc:
            norm_layer_name = bn_sc_group_layers_sc[layer]
            mu = self.orig_net.params[norm_layer_name][0].data
            var = self.orig_net.params[norm_layer_name][1].data

            eps = group_layers_bn_eps[layer]
            sigma = np.sqrt(var + eps)
            gamma = self.orig_net.params[layer][0].data
            beta = self.orig_net.params[layer][1].data
            gamma_new = gamma / sigma
            beta_new = beta - gamma * mu / sigma

            scale_new_w[layer] = gamma_new
            scale_new_b[layer] = beta_new

        # Reload the net with the updated proto
        #
        # Note: as we changed the proto topology, we need to reload the model with the updated proto to
        #        reflect the update in the model also (The unused param's in the .model will be automaticly
        #        removed.
        self.net = caffe.Net(self.updated_proto, self.model, caffe.TEST)

        # update the conv's weights and bias for the conv-batchnorm-scale group
        for layer in conv_new_w:
            self.net.params[layer + '_m'][0].data[...] = conv_new_w[layer]
            self.net.params[layer + '_m'][1].data[...] = conv_new_b[layer]

        # udpate the scale's weights and bias for the batchnorm-scale group
        for layer in scale_new_w:
            self.net.params[layer + '_f'][0].data[...] = scale_new_w[layer]
            self.net.params[layer + '_f'][1].data[...] = scale_new_b[layer]

        self.net.save(self.updated_model)
        self.net = caffe.Net(self.updated_proto, self.updated_model, caffe.TEST)
    #
    # Remove the invalid layers
    #
    def remove_invalid_layer(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(open(self.updated_proto).read(), proto)
        invalid_layer_list = []
        for i in range(len(proto.layer)):
            layer = proto.layer[i]
            if str(layer.type).lower() in ['pooling']:
                pooling_param = layer.pooling_param
                if pooling_param.round_mode == 1:
                    # floor
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print("! Please manually change the layer({})'s 'round_mode: FLOOR' to 'ceil_mode: false' from the prototxt !".format(layer.name))
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                #else:
                    #print("Notice: please manually remove the {}'s round_mode: CEIL if it is declared in the prototxt".format(layer.name))
            if str(layer.type).lower() in ['dropout']:
                invalid_layer_list.append((i,layer))
        for item in invalid_layer_list:
            (idx,layer) = item
            #print("#invalid layer: [{}] [{}] [{}] [{}]".format(idx, layer.type, layer.name, layer.top[0]))
            top_blobs = len(proto.layer[idx].top)
            while top_blobs > 0:
                top_blobs -= 1
                top_blob_name = proto.layer[idx].top[top_blobs]
                consumer_layer_list = self.get_blob_consumer_layer(proto, top_blob_name, layer.name)
                new_bottom_name = proto.layer[idx].bottom[0]

                for item in consumer_layer_list:
                    (consumer_idx, consumer_layer) = item
                    #print("  #consumer layer: [{}] [{}] [{}] [{}]".format(consumer_idx, consumer_layer.type, consumer_layer.name, len(consumer_layer.bottom)))
                    consumer_bottom_blobs = len(consumer_layer.bottom)
                    while consumer_bottom_blobs > 0:
                        consumer_bottom_blobs -= 1
                        if consumer_layer.bottom[consumer_bottom_blobs] == top_blob_name:
                            #print("    #update consumer layer bottom [{}] from [{}] to [{}]".format(consumer_bottom_blobs, proto.layer[consumer_idx].bottom[consumer_bottom_blobs], new_bottom_name))
                            proto.layer[consumer_idx].bottom[consumer_bottom_blobs] = new_bottom_name

        # remove the BN and Scale in the Conv-BN-Scale group
        for item in invalid_layer_list:
            (idx,layer) = item
            proto.layer.remove(layer)

        # save the updated network topology .proto
        with open(self.updated_proto, 'w') as f:
            f.write(str(proto))

        self.net.save(self.updated_model)
        self.net = caffe.Net(self.updated_proto, self.updated_model, caffe.TEST)
        self.net.save(self.updated_model)

    def test(self):
        np.random.seed()
        input_shape = self.net.blobs['data'].data.shape
        input_c = input_shape[1]
        input_h = input_shape[2]
        input_w = input_shape[3]
        print("Input:{}".format(self.net.blobs['data'].data.shape))
        rand_image = np.random.rand(1, input_c, input_h, input_w) * 255
        self.net.blobs['data'].data[...] = rand_image
        self.orig_net.blobs['data'].data[...] = rand_image

        # compute
        out = self.net.forward()
        orig_out = self.orig_net.forward()

        # predicted predicted class
        print("out:{}".format(out))
        print("orig_out:{}".format(orig_out))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Caffe_conv_bn_scale_fuse')
    parser.add_argument('--proto', dest='proto',
                        help="path to deploy prototxt.", type=str)
    parser.add_argument('--model', dest='model',
                        help='path to pretrained weights', type=str)
    parser.add_argument('--test', dest='test',
                        help='test the fused model', type=int, default=1)
    args = parser.parse_args()

    iCaffeBatchnormFuse = CaffeBatchnormFuse(args.proto, args.model)
    iCaffeBatchnormFuse.fuse()
    iCaffeBatchnormFuse.remove_invalid_layer()

    if args.test:
        iCaffeBatchnormFuse.test()
