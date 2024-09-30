--TEST--
NDArray::append
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([1, 2, 3, 4]);
$b = nd::array([5, 6, 7, 8]);

print_r(nd::append($a, $b)->toArray());
print_r(nd::append($a, $a)->toArray());
print_r(nd::append([1, 2, 3, 4], [1, 2, 3, 4])->toArray());
?>
--EXPECT--
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
)
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
    [4] => 1
    [5] => 2
    [6] => 3
    [7] => 4
)
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
    [4] => 1
    [5] => 2
    [6] => 3
    [7] => 4
)