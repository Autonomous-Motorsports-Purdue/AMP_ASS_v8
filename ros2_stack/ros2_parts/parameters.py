"""Parameter helper shared by the nodes."""

from rcl_interfaces.msg import ParameterDescriptor


def declare(node, name, default, description, read_only=True):
    """Declare a described parameter and return its value.

    The type of default fixes the parameter type, so pass 1.0 and not 1 for a
    float. Most of these open hardware in __init__, so they are read only.
    """
    descriptor = ParameterDescriptor(description=description, read_only=read_only)
    return node.declare_parameter(name, default, descriptor).value
