<h1 style="color:lightgreen"> How to Handle the Script File </h1>

<p> There are 3 different cases depending on what you want to do:</p>

<h2> 1. Add a Class that Inherits from another already in the Framework </h2>
<p> Nothing. </p>

<h2> 2. Add a Class that doesn't Inherit from any Class already in the Framework </h2>
<p> Go to <B>"runnable_instances.py"</B> and add the class to the <B>base_classes</B> list that will be at
the top of the file. </p>

<h2> 3. Add a Class as a standalone Module that can be used directly from the Script File </h2>
<p> Go to <B>"script_handling.py"</B> in the <B>orange_cb_recsys.script</B> package and add a new class that inherits from the 
<B>Run</B> class (or from <B>NeedsSerializationRun</B> if the module returns something that has to be stored locally). 
Implement the <I>get_associated_class</I> method (and the <I>serialize_results</I> method in the case of NeedsSerializationRun). 
<I>get_associated_class</I> should return the class associated with the run configuration (so the class you want to add). 
(<I>serialize_results</I> should configure what happens when the results for some method of the class need to be serialized).</p>

<h1 style="color:lightgreen"> FAQ </h1>

<h2> Q: Why handling some classes as standalone modules that can be used in the script file and others just as
classes that can be used in the parameters of said modules? Wouldn't it be better to create a generic run
that handles everything? </h2>
<p> A: Ideally yes. The problem is that some of the run configurations (for example the one for the RecSys) need to do 
something else besides what is already programmed. For example, the RecSys needs to serialize the outputs generated from
the RecSys run. In order to do that, some additional code needs to be added to the script handling of the RecSys module.
Ideally, in a future version of the framework where a <B>Serialization</B> module exists, this whole process can be simplified.</p>

<h2> Q: I have followed the procedure for case 2 but the class wasn't added to the dictionary, why is that? </h2>
<p> The most likely answer is that the imports aren't configured properly. 
In order for the case 2 procedure to work, imports of the classes (and the subclasses of said classes) you want to add 
to the script file need to be configured properly in the <B>"__init__.py"</B> file associated to the package where the classes are.</p>

<h2> Q: How to initiate a script file run from user perspective? </h2>
<p> There are two possible ways, both of them realised as functions and importable from 
<B>"orange_cb_recsys.script"</B>: <br>
<br>
    - Using the <B>script_run_standard</B> function and passing the path to the script file as argument <br>
    - Using the <B>script_run_with_classes_file</B> function and passing the path to the script file and the path to the classes file as argument <br>
<br>
In the second case, the classes file can be created by using the <B>serialize_classes</B> function importable from <B>"orange_cb_recsys.runnable_instances"</B>.<br>
The main difference is that there will be a slight performance improvement in the second case since the dictionary containing all the framework's classes 
will be retrieved from the classes file instead of being generated dynamically.<br>
(Keep in mind that in case of a framework's update the classes file will have to be generated again in order to use newly added classes)
</p>
