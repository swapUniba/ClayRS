Al momento ci sono 2 template html per effettuare prova più una copia per cambiamente chiamata prova.
Il motivo dei 2 template report_template2 e report_template3 sta nel processo di renderizzazione il primo di fatto
è più leggibile perché propone una intentazione sia dei tag html che degli statement di jinja ma quando viene
renderizzato da jinja questi spazzi creano confusione nel documento di report html che viene a crearsi rendendolo
sempre leggibile, ma più complicato da leggere o da modificare  per varie esigenze e per questo che viene usato
anche il secondo file di template che non presenta indentazione del codice html né degli statement e quando jinja
lo renderizza produce un risultato più leggibile.