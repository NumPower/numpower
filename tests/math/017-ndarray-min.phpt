--TEST--
NDArray::min
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::min($a));
print_r(\NDArray::min($a[0]));
print_r(\NDArray::min([[1],[2]]));
?>
--EXPECT--
111