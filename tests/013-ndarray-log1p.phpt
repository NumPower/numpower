--TEST--
NDArray::log1p
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::log1p($a)->toArray());
print_r(\NDArray::log1p($a[0])->toArray());
print_r(\NDArray::log1p([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0.6931471824646
            [1] => 1.0986123085022
        )

    [1] => Array
        (
            [0] => 1.3862943649292
            [1] => 1.6094379425049
        )

)
Array
(
    [0] => 0.6931471824646
    [1] => 1.0986123085022
)
Array
(
    [0] => Array
        (
            [0] => 0.6931471824646
        )

    [1] => Array
        (
            [0] => 1.0986123085022
        )

)