--TEST--
NDArray::max
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::max($a));
print_r(\NDArray::max($a[0]));
print_r(\NDArray::max([[1],[2]]));
?>
--EXPECT--
422