# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: postprocessor_config.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1apostprocessor_config.proto\"~\n\x0c\x44\x42SCANConfig\x12\x12\n\ndbscan_eps\x18\x01 \x01(\x02\x12\x1a\n\x12\x64\x62scan_min_samples\x18\x02 \x01(\x05\x12\x19\n\x11neighborhood_size\x18\x03 \x01(\x05\x12#\n\x1b\x64\x62scan_confidence_threshold\x18\x04 \x01(\x02\"\xd8\x01\n\x10\x43lusteringConfig\x12\x1a\n\x12\x63overage_threshold\x18\x01 \x01(\x02\x12#\n\x1bminimum_bounding_box_height\x18\x02 \x01(\x05\x12$\n\rdbscan_config\x18\x03 \x01(\x0b\x32\r.DBSCANConfig\x12/\n\nbbox_color\x18\x04 \x01(\x0b\x32\x1b.ClusteringConfig.BboxColor\x1a,\n\tBboxColor\x12\t\n\x01R\x18\x01 \x01(\x05\x12\t\n\x01G\x18\x02 \x01(\x05\x12\t\n\x01\x42\x18\x03 \x01(\x05\"\xe9\x01\n\x14PostprocessingConfig\x12Y\n\x1b\x63lasswise_clustering_config\x18\x01 \x03(\x0b\x32\x34.PostprocessingConfig.ClasswiseClusteringConfigEntry\x12\x11\n\tlinewidth\x18\x02 \x01(\x05\x12\x0e\n\x06stride\x18\x03 \x01(\x05\x1aS\n\x1e\x43lasswiseClusteringConfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.ClusteringConfig:\x02\x38\x01\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(
    DESCRIPTOR, 'postprocessor_config_pb2', globals())
if not _descriptor._USE_C_DESCRIPTORS:

    DESCRIPTOR._options = None
    _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY._options = None
    _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY._serialized_options = b'8\001'
    _DBSCANCONFIG._serialized_start = 30
    _DBSCANCONFIG._serialized_end = 156
    _CLUSTERINGCONFIG._serialized_start = 159
    _CLUSTERINGCONFIG._serialized_end = 375
    _CLUSTERINGCONFIG_BBOXCOLOR._serialized_start = 331
    _CLUSTERINGCONFIG_BBOXCOLOR._serialized_end = 375
    _POSTPROCESSINGCONFIG._serialized_start = 378
    _POSTPROCESSINGCONFIG._serialized_end = 611
    _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY._serialized_start = 528
    _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY._serialized_end = 611
# @@protoc_insertion_point(module_scope)
