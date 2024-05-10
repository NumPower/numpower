--TEST--
NDArray::sinc
--FILE--
<?php
$a = \NDArray::array([[-156, 150], [19, -39]]);
print_r(\NDArray::sinc($a)->toArray());
print_r(\NDArray::sinc($a[0])->toArray());
print_r(\NDArray::sinc([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 3.3665301657493E-8
            [1] => 5.1100769837831E-8
        )

    [1] => Array
        (
            [0] => -2.3833271356466E-8
            [1] => -3.3665301657493E-8
        )

)
Array
(
    [0] => 3.3665301657493E-8
    [1] => 5.1100769837831E-8
)
Array
(
    [0] => Array
        (
            [0] => 1
        )

    [1] => Array
        (
            [0] => 0.63661974668503
        )

)
