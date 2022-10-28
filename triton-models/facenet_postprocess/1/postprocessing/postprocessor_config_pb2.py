# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: postprocessor_config.proto

from google.protobuf import descriptor_pb2
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import reflection as _reflection
from google.protobuf import message as _message
from google.protobuf import descriptor as _descriptor
import sys
_b = sys.version_info[0] < 3 and (
    lambda x: x) or (
        lambda x: x.encode('latin1'))
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name='postprocessor_config.proto',
    package='',
    syntax='proto3',
    serialized_pb=_b('\n\x1apostprocessor_config.proto\"~\n\x0c\x44\x42SCANConfig\x12\x12\n\ndbscan_eps\x18\x01 \x01(\x02\x12\x1a\n\x12\x64\x62scan_min_samples\x18\x02 \x01(\x05\x12\x19\n\x11neighborhood_size\x18\x03 \x01(\x05\x12#\n\x1b\x64\x62scan_confidence_threshold\x18\x04 \x01(\x02\"\xd8\x01\n\x10\x43lusteringConfig\x12\x1a\n\x12\x63overage_threshold\x18\x01 \x01(\x02\x12#\n\x1bminimum_bounding_box_height\x18\x02 \x01(\x05\x12$\n\rdbscan_config\x18\x03 \x01(\x0b\x32\r.DBSCANConfig\x12/\n\nbbox_color\x18\x04 \x01(\x0b\x32\x1b.ClusteringConfig.BboxColor\x1a,\n\tBboxColor\x12\t\n\x01R\x18\x01 \x01(\x05\x12\t\n\x01G\x18\x02 \x01(\x05\x12\t\n\x01\x42\x18\x03 \x01(\x05\"\xe9\x01\n\x14PostprocessingConfig\x12Y\n\x1b\x63lasswise_clustering_config\x18\x01 \x03(\x0b\x32\x34.PostprocessingConfig.ClasswiseClusteringConfigEntry\x12\x11\n\tlinewidth\x18\x02 \x01(\x05\x12\x0e\n\x06stride\x18\x03 \x01(\x05\x1aS\n\x1e\x43lasswiseClusteringConfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12 \n\x05value\x18\x02 \x01(\x0b\x32\x11.ClusteringConfig:\x02\x38\x01\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)


_DBSCANCONFIG = _descriptor.Descriptor(
    name='DBSCANConfig',
    full_name='DBSCANConfig',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='dbscan_eps',
            full_name='DBSCANConfig.dbscan_eps',
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='dbscan_min_samples',
            full_name='DBSCANConfig.dbscan_min_samples',
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='neighborhood_size',
            full_name='DBSCANConfig.neighborhood_size',
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='dbscan_confidence_threshold',
            full_name='DBSCANConfig.dbscan_confidence_threshold',
            index=3,
            number=4,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=30,
    serialized_end=156,
)


_CLUSTERINGCONFIG_BBOXCOLOR = _descriptor.Descriptor(
    name='BboxColor',
    full_name='ClusteringConfig.BboxColor',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='R', full_name='ClusteringConfig.BboxColor.R', index=0,
            number=1, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='G', full_name='ClusteringConfig.BboxColor.G', index=1,
            number=2, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='B', full_name='ClusteringConfig.BboxColor.B', index=2,
            number=3, type=5, cpp_type=1, label=1,
            has_default_value=False, default_value=0,
            message_type=None, enum_type=None, containing_type=None,
            is_extension=False, extension_scope=None,
            options=None),
    ],
    extensions=[
    ],
    nested_types=[],
    enum_types=[
    ],
    options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[
    ],
    serialized_start=331,
    serialized_end=375,
)

_CLUSTERINGCONFIG = _descriptor.Descriptor(
    name='ClusteringConfig',
    full_name='ClusteringConfig',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='coverage_threshold',
            full_name='ClusteringConfig.coverage_threshold',
            index=0,
            number=1,
            type=2,
            cpp_type=6,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='minimum_bounding_box_height',
            full_name='ClusteringConfig.minimum_bounding_box_height',
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='dbscan_config',
            full_name='ClusteringConfig.dbscan_config',
            index=2,
            number=3,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='bbox_color',
            full_name='ClusteringConfig.bbox_color',
            index=3,
            number=4,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
    ],
    extensions=[],
    nested_types=[
        _CLUSTERINGCONFIG_BBOXCOLOR,
    ],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=159,
    serialized_end=375,
)


_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY = _descriptor.Descriptor(
    name='ClasswiseClusteringConfigEntry',
    full_name='PostprocessingConfig.ClasswiseClusteringConfigEntry',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='key',
            full_name='PostprocessingConfig.ClasswiseClusteringConfigEntry.key',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=_b("").decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='value',
            full_name='PostprocessingConfig.ClasswiseClusteringConfigEntry.value',
            index=1,
            number=2,
            type=11,
            cpp_type=10,
            label=1,
            has_default_value=False,
            default_value=None,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    options=_descriptor._ParseOptions(
        descriptor_pb2.MessageOptions(),
        _b('8\001')),
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=528,
    serialized_end=611,
)

_POSTPROCESSINGCONFIG = _descriptor.Descriptor(
    name='PostprocessingConfig',
    full_name='PostprocessingConfig',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    fields=[
        _descriptor.FieldDescriptor(
            name='classwise_clustering_config',
            full_name='PostprocessingConfig.classwise_clustering_config',
            index=0,
            number=1,
            type=11,
            cpp_type=10,
            label=3,
            has_default_value=False,
            default_value=[],
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='linewidth',
            full_name='PostprocessingConfig.linewidth',
            index=1,
            number=2,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
        _descriptor.FieldDescriptor(
            name='stride',
            full_name='PostprocessingConfig.stride',
            index=2,
            number=3,
            type=5,
            cpp_type=1,
            label=1,
            has_default_value=False,
            default_value=0,
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            options=None),
    ],
    extensions=[],
    nested_types=[
        _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY,
    ],
    enum_types=[],
    options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=378,
    serialized_end=611,
)

_CLUSTERINGCONFIG_BBOXCOLOR.containing_type = _CLUSTERINGCONFIG
_CLUSTERINGCONFIG.fields_by_name['dbscan_config'].message_type = _DBSCANCONFIG
_CLUSTERINGCONFIG.fields_by_name['bbox_color'].message_type = _CLUSTERINGCONFIG_BBOXCOLOR
_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY.fields_by_name[
    'value'].message_type = _CLUSTERINGCONFIG
_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY.containing_type = _POSTPROCESSINGCONFIG
_POSTPROCESSINGCONFIG.fields_by_name['classwise_clustering_config'].message_type = _POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY
DESCRIPTOR.message_types_by_name['DBSCANConfig'] = _DBSCANCONFIG
DESCRIPTOR.message_types_by_name['ClusteringConfig'] = _CLUSTERINGCONFIG
DESCRIPTOR.message_types_by_name['PostprocessingConfig'] = _POSTPROCESSINGCONFIG

DBSCANConfig = _reflection.GeneratedProtocolMessageType('DBSCANConfig', (_message.Message,), dict(
    DESCRIPTOR=_DBSCANCONFIG,
    __module__='postprocessor_config_pb2'
    # @@protoc_insertion_point(class_scope:DBSCANConfig)
))
_sym_db.RegisterMessage(DBSCANConfig)

ClusteringConfig = _reflection.GeneratedProtocolMessageType('ClusteringConfig', (_message.Message,), dict(

    BboxColor=_reflection.GeneratedProtocolMessageType('BboxColor', (_message.Message,), dict(
        DESCRIPTOR=_CLUSTERINGCONFIG_BBOXCOLOR,
        __module__='postprocessor_config_pb2'
        # @@protoc_insertion_point(class_scope:ClusteringConfig.BboxColor)
    )),
    DESCRIPTOR=_CLUSTERINGCONFIG,
    __module__='postprocessor_config_pb2'
    # @@protoc_insertion_point(class_scope:ClusteringConfig)
))
_sym_db.RegisterMessage(ClusteringConfig)
_sym_db.RegisterMessage(ClusteringConfig.BboxColor)

PostprocessingConfig = _reflection.GeneratedProtocolMessageType('PostprocessingConfig', (_message.Message,), dict(

    ClasswiseClusteringConfigEntry=_reflection.GeneratedProtocolMessageType('ClasswiseClusteringConfigEntry', (_message.Message,), dict(
        DESCRIPTOR=_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY,
        __module__='postprocessor_config_pb2'
        # @@protoc_insertion_point(class_scope:PostprocessingConfig.ClasswiseClusteringConfigEntry)
    )),
    DESCRIPTOR=_POSTPROCESSINGCONFIG,
    __module__='postprocessor_config_pb2'
    # @@protoc_insertion_point(class_scope:PostprocessingConfig)
))
_sym_db.RegisterMessage(PostprocessingConfig)
_sym_db.RegisterMessage(PostprocessingConfig.ClasswiseClusteringConfigEntry)


_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY.has_options = True
_POSTPROCESSINGCONFIG_CLASSWISECLUSTERINGCONFIGENTRY._options = _descriptor._ParseOptions(
    descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
