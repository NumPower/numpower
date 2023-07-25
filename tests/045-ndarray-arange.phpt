--TEST--
NDArray::ceil
--FILE--
<?php
$a = \NDArray::arange(20, 10, 1);
print_r($a->toArray());
?>
--EXPECT--
Array
(
    [0] => 10
    [1] => 11
    [2] => 12
    [3] => 13
    [4] => 14
    [5] => 15
    [6] => 16
    [7] => 17
    [8] => 18
    [9] => 19
)