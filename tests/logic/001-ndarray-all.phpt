--TEST--
NDArray::all
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::all($a));
print_r(\NDArray::all($a[0]));
print_r(\NDArray::all($c));
?>
--EXPECT--
011