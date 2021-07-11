<h1 style="color:lightgreen"> How to Handle the Script File </h1>

<p> There are 3 different cases depending on what you want to do:</p>

<h2> 1. Add a Class that Inherits from another already in the Framework </h2>
<p> Go to <B>"runnable_instances.py"</B> in the <B>utils</B> package and run the main of the file. This will rewrite
the file where the dictionary containing the framework's classes is stored and will update it</p>

<h2> 2. Add a Class that doesn't Inherit from any Class already in the Framework </h2>
<p> Go to <B>"runnable_instances.py"</B> and add the class to the <B>base_classes</B> list that will be at
the top of the file. After doing that, do the same as in case 1.</p>

<h2> 3. Add a Class as a standalone Module that can be used directly from the Script File </h2>
<p> Go to <B>"script_handling.py"</B> in the <B>orange_cb_recsys</B> package and add a new class that inherits from the <B>Run</B> class (or from <B>NeedsSerializationRun</B> if the module returns something that has to be stored locally). 
Implement the <I>get_associated_class</I> method (and the <I>serialize_results</I> method in the case of NeedsSerializationRun). 
<I>get_associated_class</I> should return the class associated with the run configuration (so the class you want to add). 
(<I>serialize_results</I> should configure what happens when the results for some method of the class need to be serialized)</p>

<h1 style="color:lightgreen"> FAQ </h1>

<h2> Q: Why handling some classes as standalone modules that can be used in the script file and others just as
classes that can be used in the parameters of said modules? Wouldn't it be better to create a generic run
that handles everything? </h2>
<p> A: Ideally yes. The problem is that some of the run configurations (for example the one for the RecSys) need to do something else besides what is already programmed. For example, the RecSys needs to serialize the outputs generated from
the RecSys run. In order to do that, some additional code needs to be added to the script handling of the RecSys module.
Ideally, in a future version of the framework where a <B>Serialization</B> module exists, this whole process can be simplified.</p>

<h2> Q: I have followed the procedure for case 2 but the class wasn't added to the dictionary, why is that? </h2>
<p> The most likely answer is that the imports aren't configured properly. 
In order for the case 2 procedure to work, imports of the classes (and the subclasses of said classes) you want to add to the script file need to be configured properly in the <B>"__init__.py"</B> file associated to the package where the classes are.</p>
