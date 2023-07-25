--TEST--
NDArray::allclose
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
print_r(\NDArray::allclose($a, $b));
print_r(\NDArray::allclose($a, $a));
print_r(\NDArray::allclose($c, $c));
?>
--EXPECT--
11