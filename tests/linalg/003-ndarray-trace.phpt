--TEST--
NDArray::trace
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
$d = \NDArray::array([[9], [10]]);
print_r(\NDArray::trace($a));
print_r(\NDArray::trace([[1, 2], [3, 4]]));
print_r(\NDArray::trace($d));
?>
--EXPECT--
559