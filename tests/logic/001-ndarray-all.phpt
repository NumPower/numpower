--TEST--
NDArray::all
--FILE--
<?php
$a = \NDArray::array([[1, 0], [3, 4]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::all($a));
print_r(\NDArray::all($a[0]));
print_r(\NDArray::all($c));
?>
--EXPECT--
001