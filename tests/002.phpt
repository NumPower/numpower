--TEST--
test1() Basic test
--EXTENSIONS--
phpsci_ndarray
--FILE--
<?php
$ret = test1();

var_dump($ret);
?>
--EXPECT--
The extension phpsci_ndarray is loaded and working!
NULL
