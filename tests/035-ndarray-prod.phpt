--TEST--
NDArray::prod
--FILE--
<?php
$a = \NDArray::array([[-156.50, 150.525435], [0, -39.151414]]);
print_r(\NDArray::prod($a));
print_r(\NDArray::prod($a, axis: 0)->toArray());
print_r(\NDArray::prod($a, axis: 1)->toArray());
print_r(\NDArray::prod($a[0]));
print_r(\NDArray::prod([[0.12],[-0.513124]]));
?>
--EXPECT--
0Array
(
    [0] => -0
    [1] => -5893.2836914062
)
Array
(
    [0] => -23557.23046875
    [1] => -0
)
-23557.23046875-0.061574876308441