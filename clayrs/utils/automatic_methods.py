import inspect


def autorepr(obj, frame):
    # pull tuple from frame
    args, args_paramname, kwargs_paramname, values = inspect.getargvalues(frame)

    args = args[1:]  # remove 'self' argument from function

    arg_string = ''

    # add to arg string formal argument
    arg_string += ', '.join([f"{arg}={repr(values[arg])}"
                             for arg in (args if args is not None else [])])

    # show positional varargs
    if args_paramname is not None:
        varglist = values[args_paramname]
        if len(arg_string) != 0:
            arg_string += ', '
        arg_string += ', '.join([f"*{args_paramname}={repr(v)}"
                                 for v in (varglist if varglist is not None else [])])

    # show named varargs
    if kwargs_paramname is not None:
        varglist = values[kwargs_paramname]
        if len(arg_string) != 0:
            arg_string += ', '
        arg_string += ', '.join([f"*{kwargs_paramname}_{k}={repr(varglist[k])}"
                                 for k in (sorted(varglist) if varglist is not None else [])])

    name_obj = obj.__class__.__name__
    repr_string = f"{name_obj}({arg_string})"

    return repr_string
