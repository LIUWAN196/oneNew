#ifndef __LOG_H__
#define __LOG_H__

#include <stdio.h>
#include <stdlib.h>

// 1、定义几个日志等级，用于打印信息和调试
typedef enum {
    NO_LOG_LEVEL = 0,  // 不会有任何日志打印
    ERR_LEVEL = 1,
    DBG_LEVEL = 2,
    MSG_LEVEL = 3
} LOG_LEVEL;

//// 暂时手动定义日志等级为 MSG_LEVEL 级别，便于调试
//#define LOG_LEV MSG_LEVEL

// 3、打印调试信息
#ifdef LOG_LEV

#define log_printf(file, func, line, level, fmt, ...)                                                            \
    do {                                                                                                         \
        if (level > LOG_LEV) {                                                                                   \
            break;                                                                                               \
        }                                                                                                        \
        switch (level) {                                                                                         \
            case ERR_LEVEL:                                                                                      \
                printf("\033[31m[%s][ERROR]: %s:%d: Error content: " fmt "\033[0m\n", __TIME__, file, line,      \
                       ##__VA_ARGS__);                                                                           \
                printf("\033[31m[%s][EXIT PROGRAM]: have entered the LOG_ERR branch, please using above Error "  \
                       "content to debug\033[0m\n",                                                              \
                       __TIME__);                                                                                \
                abort();                                                                                         \
                break;                                                                                           \
            case DBG_LEVEL:                                                                                      \
                printf("\033[33m[%s][DEBUG]: %s:%d: Debug content: " fmt "\033[0m\n", __TIME__, file, line,      \
                       ##__VA_ARGS__);                                                                           \
                break;                                                                                           \
            case MSG_LEVEL:                                                                                      \
                printf("[%s][MESSAGE]: %s:%d: Message content: " fmt "\n", __TIME__, file, line, ##__VA_ARGS__); \
                break;                                                                                           \
            default:                                                                                             \
                break;                                                                                           \
        }                                                                                                        \
    } while (0)
#else
// 不做任何操作
#define CHECK(info)
#define log_printf(file, func, line, level, fmt, ...)
#endif

#define LOG_ERR(fmt, ...) log_printf(__FILE__, __FUNCTION__, __LINE__, ERR_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_DBG(fmt, ...) log_printf(__FILE__, __FUNCTION__, __LINE__, DBG_LEVEL, fmt, ##__VA_ARGS__)
#define LOG_MSG(fmt, ...) log_printf(__FILE__, __FUNCTION__, __LINE__, MSG_LEVEL, fmt, ##__VA_ARGS__)

#endif
