--TEST--
NDArray::log2
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::log2($a)->toArray());
print_r(\NDArray::log2($a[0])->toArray());
print_r(\NDArray::log2([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1.5849624872208
            [1] => 2
        )

)
Array
(
    [0] => 0
    [1] => 1
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 1
        )

)