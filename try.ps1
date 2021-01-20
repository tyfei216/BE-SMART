Get-ChildItem ../Libps/ | ForEach-Object -Process{
    if($_ -is [System.IO.FileInfo])
    {
        $b = $_
        $a = -Join("../Libps/", $_.name)
        echo $a
        if ($b.name.Contains("BE3")) {
            python ./train.py -gpu -ds $a -baseIndex 1 -tensorboard -evalpositions 2 3 4 5 6 7 8 9 10
        }
        elseif ($b.name.Contains("BE4")) {
            python ./train.py -gpu -ds $a -baseIndex 1 -tensorboard -evalpositions 2 3 4 5 6 7 8 9 10
        }
        else {
            python ./train.py -gpu -ds $a -baseIndex 2 -tensorboard
        } 
    }
}