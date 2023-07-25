--TEST--
NDArray::log
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::log($a)->toArray());
print_r(\NDArray::log($a[0])->toArray());
print_r(\NDArray::log([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0.6931471824646
        )

    [1] => Array
        (
            [0] => 1.0986123085022
            [1] => 1.3862943649292
        )

)
Array
(
    [0] => 0
    [1] => 0.6931471824646
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 0.6931471824646
        )

)