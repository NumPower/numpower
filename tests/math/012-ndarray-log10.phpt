--TEST--
NDArray::log10
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::log10($a)->toArray());
print_r(\NDArray::log10($a[0])->toArray());
print_r(\NDArray::log10([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0.30103000998497
        )

    [1] => Array
        (
            [0] => 0.47712123394012
            [1] => 0.60206001996994
        )

)
Array
(
    [0] => 0
    [1] => 0.30103000998497
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 0.30103000998497
        )

)