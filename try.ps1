Get-ChildItem ../Libps/ | ForEach-Object -Process{
    if($_ -is [System.IO.FileInfo])
    {
        $b = $_
        $a = -Join("../Libps/", $_.name)
        echo $a
        if ($b.name.Contains("BE3")) {
            python ./train.py -gpu -ds $a -baseIndex 1 -tensorboard
        }
        elseif ($b.name.Contains("BE4")) {
            python ./train.py -gpu -ds $a -baseIndex 1 -tensorboard
        }
        else {
            python ./train.py -gpu -ds $a -baseIndex 2 -tensorboard
        } 
    }
}