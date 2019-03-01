# register datastore
az ml datastore register-blob -n $datastorename -a $accountname -k $accountkey -c $containername

# set default datastore
az ml datastore set-default -n $datastorename