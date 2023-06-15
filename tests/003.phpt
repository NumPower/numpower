--TEST--
test2() Basic test
--EXTENSIONS--
phpsci_ndarray
--FILE--
<?php
var_dump(test2());
var_dump(test2('PHP'));
?>
--EXPECT--
string(11) "Hello World"
string(9) "Hello PHP"
