--TEST--
NDArray::exp
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::exp($a)->toArray());
print_r(\NDArray::exp($a[0])->toArray());
print_r(\NDArray::exp([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 2.7182817459106
            [1] => 7.3890562057495
        )

    [1] => Array
        (
            [0] => 20.085536956787
            [1] => 54.598148345947
        )

)
Array
(
    [0] => 2.7182817459106
    [1] => 7.3890562057495
)
Array
(
    [0] => Array
        (
            [0] => 2.7182817459106
        )

    [1] => Array
        (
            [0] => 7.3890562057495
        )

)