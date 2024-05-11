--TEST--
NDArray::arange
--FILE--
<?php
$a = \NDArray::arange(20, 10, 1);
print_r($a->toArray());

$b = \NDArray::arange(10);
print_r($b->toArray());

$c = \NDArray::arange(10, 1);
print_r($c->toArray());
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
Array
(
    [0] => 0
    [1] => 1
    [2] => 2
    [3] => 3
    [4] => 4
    [5] => 5
    [6] => 6
    [7] => 7
    [8] => 8
    [9] => 9
)
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
    [4] => 5
    [5] => 6
    [6] => 7
    [7] => 8
    [8] => 9
)