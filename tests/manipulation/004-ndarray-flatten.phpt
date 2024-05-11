--TEST--
NDArray::flatten
--FILE--
<?php
use \NDArray as nd;

$a = nd::array([[1, 2, 3, 4]]);
$b = nd::array([[5, 6], [7, 8]]);
$c = nd::array([[[5, 6], [7, 8]], [[5, 6], [7, 8]]]);
$d = nd::array([1, 2, 3, 4]);

print_r(nd::flatten($a)->toArray());
print_r(nd::flatten($b)->toArray());
print_r(nd::flatten($c)->toArray());
print_r(nd::flatten($d)->toArray());
print_r(nd::flatten([[5, 6], [7, 8]])->toArray());
?>
--EXPECT--
Array
(
    [0] => 1
    [1] => 2
    [2] => 3
    [3] => 4
)
Array
(
    [0] => 5
    [1] => 6
    [2] => 7
    [3] => 8
)
Array
(
    [0] => 5
    [1] => 6
    [2] => 7
    [3] => 8
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
)
Array
(
    [0] => 5
    [1] => 6
    [2] => 7
    [3] => 8
)